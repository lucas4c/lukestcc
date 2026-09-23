from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json
import pandas as pd
import re
import time
import psutil

# Tentativa de carregar biblioteca NVML para leitura de VRAM (GPUs NVIDIA)
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Pega a GPU primária (index 0)
except ImportError:
    GPU_AVAILABLE = False
    print("[AVISO] Biblioteca 'pynvml' não instalada. VRAM não será medida.")
except pynvml.NVMLError:
    GPU_AVAILABLE = False
    print("[AVISO] GPU NVIDIA não detectada ou driver ausente. VRAM não será medida.")

# ==================================================
# CONFIGURAÇÃO DO MODELO
# ==================================================
MODEL_NAME = "llama3.2:1b"

model = ChatOllama(
    model=MODEL_NAME,
    temperature=0.0
)

# ==================================================
# PROMPT SECO (BASELINE)
# ==================================================
template = """
You are evaluating a dialogue for the character {npc_name} in the game Skyrim.

Player says:
"{player_message}"

Which response is the most appropriate?

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Respond with ONLY one letter: A, B, C or D.
Do not provide any further explanation, only respond with the most appropriate option.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# ==================================================
# CARREGAR DATASET
# ==================================================
dataset_path = "datasets/skyrim_benchmark_full.json"
try:
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
except FileNotFoundError:
    print(f"[ERRO] Arquivo {dataset_path} não encontrado.")
    exit()

# ==================================================
# FUNÇÕES DE MONITORAMENTO
# ==================================================
def get_ram_usage():
    """Retorna o uso atual de RAM pelo processo (em MB)"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def get_vram_usage():
    """Retorna o uso atual de VRAM da GPU 0 (em MB)"""
    if GPU_AVAILABLE:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            return info.used / (1024 * 1024)
        except Exception:
            return 0.0
    return 0.0

# ==================================================
# LOOP DE AVALIAÇÃO
# ==================================================
results = []
correct = 0
total = 0

print(f"===================================")
print(f" INICIANDO BENCHMARK BASELINE ")
print(f" Modelo: {MODEL_NAME}")
print(f"===================================")

# Medição de tempo global
global_start_time = time.time()

for npc in dataset["characters"]:
    npc_name = npc["character"]
    print(f"\n--- Avaliando NPC: {npc_name} ---")

    for test in npc["dialogues"]:
        
        # Medições pré-inferência
        step_start_time = time.time()
        ram_before = get_ram_usage()
        vram_before = get_vram_usage()

        # Invoca o modelo
        result = chain.invoke({
            "npc_name": npc_name,
            "player_message": test["player_message"],
            "option_a": test["options"]["A"],
            "option_b": test["options"]["B"],
            "option_c": test["options"]["C"],
            "option_d": test["options"]["D"]
        })

        # Medições pós-inferência
        step_end_time = time.time()
        ram_after = get_ram_usage()
        vram_after = get_vram_usage()

        # Cálculo de métricas
        latency_seconds = step_end_time - step_start_time
        ram_delta = ram_after - ram_before
        vram_delta = vram_after - vram_before

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
            "npc": npc_name,
            "step": test["step"],
            "expected": expected,
            "predicted": prediction,
            "raw_output": raw_prediction,
            "correct": is_correct,
            # Métricas anexadas ao CSV
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

# Calcula médias
avg_latency = sum(r['latency_sec'] for r in results) / len(results)
avg_vram = sum(r['vram_used_mb'] for r in results) / len(results)

print("\n===================================")
print(" RESULTADOS FINAIS (BASELINE)")
print("===================================")
print(f"Modelo:       {MODEL_NAME}")
print(f"Corretas:     {correct}/{total} ({accuracy:.2%})")
print(f"Tempo Total:  {total_duration_sec:.2f} segundos")
print(f"Média Latência: {avg_latency:.2f}s por requisição")
if GPU_AVAILABLE:
    print(f"Média VRAM:   {avg_vram:.0f} MB")

# Cria uma string segura para o nome do arquivo, removendo caracteres inválidos no Windows
safe_model_name = re.sub(r'[\\/*?:"<>|]', "_", MODEL_NAME)

csv_filename = f"phase1_baseline_results_{safe_model_name}_metrics.csv"
pd.DataFrame(results).to_csv(csv_filename, index=False)
print(f"\nResultados e métricas salvos em: {csv_filename}")

# Finaliza driver da GPU
if GPU_AVAILABLE:
    pynvml.nvmlShutdown()