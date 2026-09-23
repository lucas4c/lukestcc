from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json
import pandas as pd
import re
import time
import psutil

# ==================================================
# CONFIGURAÇÃO DE VRAM
# ==================================================
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    GPU_AVAILABLE = False
    print("[AVISO] Medição de VRAM desativada (pynvml não disponível ou falhou).")

def get_ram_usage():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def get_vram_usage():
    if GPU_AVAILABLE:
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024 * 1024)
        except Exception:
            pass
    return 0.0

# ==================================================
# CONFIGURAÇÃO DO MODELO
# ==================================================
MODEL_NAME = "llama3.2:1b"

model = ChatOllama(
    model=MODEL_NAME,
    temperature=0.0
)

# ==================================================
# CARREGAR PERSONAS (JSONL)
# ==================================================
# Carrega o JSONL e cria um dicionário indexado pelo nome do personagem em letras minúsculas
personas_db = {}
jsonl_path = "datasets/Skyrim_characters_structured.jsonl"
try:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                char_data = json.loads(line)
                # O nome no JSONL é 'entity_name'
                name_key = char_data.get("entity_name", "").lower()
                personas_db[name_key] = char_data
    print(f"[OK] Carregadas {len(personas_db)} personas do arquivo JSONL.")
except FileNotFoundError:
    print(f"[ERRO] Arquivo {jsonl_path} não encontrado.")
    exit()

# ==================================================
# PROMPT DA FASE 2 (CONTEXTO ESTÁTICO / PERSONA)
# ==================================================
template = """
You are evaluating a dialogue in the game Skyrim. 
You must think and respond as the following character:

Name: {npc_name}
Race: {race}
Gender: {gender}
Occupation: {occupation}
Home City: {home_city}
Morality: {morality}
Aggression: {aggression}

Maintain the tone, personality, and knowledge appropriate for this character.

Player says:
"{player_message}"

Which response is the most appropriate and in-character for you to say next?

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Respond with ONLY one letter: A, B, C or D. Do not explain your reasoning.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# ==================================================
# CARREGAR DATASET DE TESTE
# ==================================================
dataset_path = "datasets/skyrim_benchmark_full.json"
try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
except FileNotFoundError:
    print(f"[ERRO] Arquivo {dataset_path} não encontrado.")
    exit()

# ==================================================
# LOOP DE AVALIAÇÃO
# ==================================================
results = []
correct = 0
total = 0

print(f"\n===================================")
print(f" INICIANDO FASE 2: CONTEXTO ESTÁTICO ")
print(f" Modelo: {MODEL_NAME}")
print(f"===================================")

global_start_time = time.time()

for npc in dataset["characters"]:
    npc_name = npc["character"]
    
    # Busca a persona no dicionário carregado do JSONL
    persona = personas_db.get(npc_name.lower(), {})
    
    # Se o personagem não existir no JSONL, preenchemos com "Unknown" para o teste não quebrar
    race = persona.get("race", "Unknown")
    gender = persona.get("gender", "Unknown")
    occupation = persona.get("occupation", "Unknown")
    home_city = persona.get("home_city", "Unknown")
    morality = persona.get("morality", "Unknown")
    aggression = persona.get("aggression", "Unknown")

    print(f"\n--- Avaliando NPC: {npc_name} ({occupation} - {home_city}) ---")

    for test in npc["dialogues"]:
        
        step_start_time = time.time()
        ram_before = get_ram_usage()
        vram_before = get_vram_usage()

        # Invoca a cadeia com os dados da Persona
        result = chain.invoke({
            "npc_name": npc_name,
            "race": race,
            "gender": gender,
            "occupation": occupation,
            "home_city": home_city,
            "morality": morality,
            "aggression": aggression,
            "player_message": test["player_message"],
            "option_a": test["options"]["A"],
            "option_b": test["options"]["B"],
            "option_c": test["options"]["C"],
            "option_d": test["options"]["D"]
        })

        step_end_time = time.time()
        ram_after = get_ram_usage()
        vram_after = get_vram_usage()

        latency_seconds = step_end_time - step_start_time

        # Processamento da resposta
        raw_prediction = result.content.strip().upper()
        match = re.search(r'\b[ABCD]\b', raw_prediction)
        prediction = match.group(0) if match else "X"

        expected = test["correct"]
        is_correct = prediction == expected

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "model": MODEL_NAME,
            "phase": "Static Context",
            "npc": npc_name,
            "step": test["step"],
            "expected": expected,
            "predicted": prediction,
            "raw_output": raw_prediction,
            "correct": is_correct,
            "latency_sec": round(latency_seconds, 2),
            "ram_used_mb": round(ram_after, 2),
            "vram_used_mb": round(vram_after, 2)
        })

        print(f"Step {test['step']} | Pred={prediction} {'✓' if is_correct else '✗'} "
              f"| Latency: {latency_seconds:.2f}s | VRAM: {vram_after:.0f}MB")

# ==================================================
# RESULTADOS E EXPORTAÇÃO
# ==================================================
global_end_time = time.time()
total_duration_sec = global_end_time - global_start_time
accuracy = correct / total if total else 0

avg_latency = sum(r['latency_sec'] for r in results) / len(results) if results else 0
avg_vram = sum(r['vram_used_mb'] for r in results) / len(results) if results else 0

print("\n===================================")
print(" RESULTADOS FINAIS (FASE 2)")
print("===================================")
print(f"Modelo:       {MODEL_NAME}")
print(f"Corretas:     {correct}/{total} ({accuracy:.2%})")
print(f"Tempo Total:  {total_duration_sec:.2f} segundos")
print(f"Média Latência: {avg_latency:.2f}s por requisição")
if GPU_AVAILABLE:
    print(f"Média VRAM:   {avg_vram:.0f} MB")

# Cria uma string segura para o nome do arquivo, removendo caracteres inválidos no Windows
safe_model_name = re.sub(r'[\\/*?:"<>|]', "_", MODEL_NAME)

csv_filename = f"phase2_static_context_{safe_model_name}_metrics.csv"
pd.DataFrame(results).to_csv(csv_filename, index=False)
print(f"\nResultados e métricas salvos em: {csv_filename}")

if GPU_AVAILABLE:
    pynvml.nvmlShutdown()