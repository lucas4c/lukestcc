from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

import json
import pandas as pd

# ==================================================
# MODEL
# ==================================================

# Utilize o nome exato do modelo baixado no Ollama
model = ChatOllama(
    model="tinyllama", # ou "llama3.2:1b" / "qwen2.5:0.5b"
    temperature=0.0
)

# ==================================================
# PROMPT
# ==================================================

template = """
You are evaluating a dialogue.

Player says:

{player_message}

Which response is the most appropriate?

A) {option_a}

B) {option_b}

C) {option_c}

D) {option_d}

Respond with ONLY one letter:
A, B, C or D.
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

# ==================================================
# LOAD DATASET
# ==================================================

with open(
    "datasets/skyrim_benchmark_full.json",
    "r",
    encoding="utf-8"
) as f:
    dataset = json.load(f)

# ==================================================
# TEST LOOP
# ==================================================

results = []
correct = 0
total = 0

for npc in dataset["characters"]:
    npc_name = npc["character"]

    print()
    print("===================================")
    print(f"NPC: {npc_name}")
    print("===================================")

    for test in npc["dialogues"]:
        result = chain.invoke({
            "player_message": test["player_message"],
            "option_a": test["options"]["A"],
            "option_b": test["options"]["B"],
            "option_c": test["options"]["C"],
            "option_d": test["options"]["D"]
        })

        # Modelos locais menores podem às vezes adicionar pontuação extra ou palavras.
        # Pegar apenas o primeiro caractere válido ajuda a limpar a saída.
        raw_prediction = result.content.strip().upper()
        prediction = raw_prediction[0] if raw_prediction else ""

        expected = test["correct"]
        is_correct = prediction == expected

        if is_correct:
            correct += 1

        total += 1

        results.append({
            "npc": npc_name,
            "step": test["step"],
            "expected": expected,
            "predicted": raw_prediction, # Salva o output cru para você debugar se o modelo for verboso
            "correct": is_correct
        })

        print(
            f"Step {test['step']} | "
            f"Expected={expected} "
            f"Predicted={prediction} "
            f"{'✓' if is_correct else '✗'}"
        )

# ==================================================
# RESULTS
# ==================================================

accuracy = correct / total if total else 0

print()
print("===================================")
print("FINAL RESULTS")
print("===================================")
print(f"Correct:  {correct}")
print(f"Total:    {total}")
print(f"Accuracy: {accuracy:.2%}")

# ==================================================
# SAVE CSV
# ==================================================

pd.DataFrame(results).to_csv(
    "baseline_results.csv",
    index=False
)

print()
print("Saved baseline_results.csv")