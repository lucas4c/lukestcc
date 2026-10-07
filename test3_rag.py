from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json
import pandas as pd
import re
import time
import psutil
from vector import retriever # Importa o seu banco vetorial (ChromaDB)

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
MODEL_NAME = "llama3.2:latest"

# Cria um nome seguro para salvar o arquivo no Windows
safe_model_name = re.sub(r'[\\/*?:"<>|]', "_", MODEL_NAME)

model = ChatOllama(
    model=MODEL_NAME,
    temperature=0.0
)

# ==================================================
# CARREGAR PERSONAS (JSONL)
# ==================================================
personas_db = {}
jsonl_path = "datasets/Skyrim_characters_structured.jsonl"
try:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                char_data = json.loads(line)
                name_key = char_data.get("entity_name", "").lower()
                personas_db[name_key] = char_data
except FileNotFoundError:
    print(f"[ERRO] Arquivo {jsonl_path} não encontrado.")
    exit()

# ==================================================
# PROMPT DA FASE 3 (PERSONA + RAG)
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

Here is relevant world knowledge from your database:
{rag_context}

Maintain the tone, personality, and knowledge appropriate for this character.

Player says:
"{player_message}"

Which response is the most appropriate and in-character for you to say next?

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Read all four options carefully before answering. Which option (A, B, C, or D) is the most accurate response? Respond with ONLY a single letter. Do not explain.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# ==================================================
# CARREGAR DATASET DE TESTE
# ==================================================
dataset_path = "datasets/skyrim_benchmark_full_new.json"
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
print(f" INICIANDO FASE 3: PERSONA + RAG ")
print(f" Modelo: {MODEL_NAME}")
print(f"===================================")

global_start_time = time.time()

for npc in dataset["characters"]:
    npc_name = npc["character"]
    persona = personas_db.get(npc_name.lower(), {})
    
    race = persona.get("race", "Unknown")
    gender = persona.get("gender", "Unknown")
    occupation = persona.get("occupation", "Unknown")
    home_city = persona.get("home_city", "Unknown")
    morality = persona.get("morality", "Unknown")
    aggression = persona.get("aggression", "Unknown")

    print(f"\n--- Avaliando NPC: {npc_name} ---")

    for test in npc["dialogues"]:
        
        step_start_time = time.time()
        ram_before = get_ram_usage()
        vram_before = get_vram_usage()

        # 1. Busca Vetorial (RAG)
        # Usamos a frase do jogador como query para buscar na Wiki do jogo
        # Query Transformation: Adicionamos o nome do NPC e localização para forçar 
        # o banco vetorial a buscar o assunto (Civil War) vinculado àquela entidade.
        rag_query = f"{npc_name} from {home_city}. {test['player_message']}"

        try:
            retrieved_docs = retriever.invoke(rag_query)
            rag_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as e:
            rag_text = "No relevant knowledge found."
            
        # Limita o tamanho do texto do RAG para não estourar a janela de contexto de modelos menores
        if len(rag_text) > 2000:
            rag_text = rag_text[:2000] + "... [truncado]"

        # 1. Primeiro, gere a string final do prompt preenchida com as variáveis
        prompt_enviado = prompt.format(
            npc_name=npc_name,
            race=race,
            gender=gender,
            occupation=occupation,
            home_city=home_city,
            morality=morality,
            aggression=aggression,
            rag_context=rag_text,
            player_message=test["player_message"],
            option_a=test["options"]["A"],
            option_b=test["options"]["B"],
            option_c=test["options"]["C"],
            option_d=test["options"]["D"]
        )

        # Agora a variável `prompt_enviado` contém a string exata que o LLM vai ler.
        # Você pode dar print() ou salvá-la no dicionário de resultados.

        # 2. Invoque o modelo passando a string formatada
        result = model.invoke(prompt_enviado)

        prompt_enviado.strip().upper();

        step_end_time = time.time()
        ram_after = get_ram_usage()
        vram_after = get_vram_usage()

        latency_seconds = step_end_time - step_start_time

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
            "phase": "RAG Context",
            "npc": npc_name,
            "step": test["step"],
            "expected": expected,
            "prompt_enviado": prompt_enviado,
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
print(" RESULTADOS FINAIS (FASE 3)")
print("===================================")
print(f"Modelo:       {MODEL_NAME}")
print(f"Corretas:     {correct}/{total} ({accuracy:.2%})")
print(f"Tempo Total:  {total_duration_sec:.2f} segundos")
print(f"Média Latência: {avg_latency:.2f}s por requisição")
if GPU_AVAILABLE:
    print(f"Média VRAM:   {avg_vram:.0f} MB")

# Salva usando a variável segura
csv_filename = f"phase3_rag_{safe_model_name}_metrics.csv"
pd.DataFrame(results).to_csv(csv_filename, index=False)
print(f"\nResultados e métricas salvos em: {csv_filename}")

if GPU_AVAILABLE:
    pynvml.nvmlShutdown()