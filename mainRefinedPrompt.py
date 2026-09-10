from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import json
import re
import pandas as pd

# ==================================================
# MODEL
# ==================================================

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct"
)

# ==================================================
# PROMPT
# ==================================================

template = """
You are roleplaying a Skyrim NPC.

Character Sheet:

Race: {race}
Gender: {gender}
Occupation: {occupation}
Home City: {home_city}
Location: {location}
Morality: {morality}
Aggression: {aggression}

The player says:

{player_message}

Which answer is most likely to be spoken by this character?

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Return ONLY:
A
B
C
or D
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

# ==================================================
# LOAD BENCHMARK
# ==================================================

with open(
    "datasets/dialogue_options.json",
    "r",
    encoding="utf-8"
) as f:

    benchmark = json.load(f)

# ==================================================
# LOAD KNOWLEDGE BASE
# ==================================================

knowledge = {}

with open(
    "datasets/Skyrim_characters_structured.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        doc = json.loads(line)

        if doc["entity_type"] != "character":
            continue

        knowledge[
            doc["entity_name"].lower()
        ] = doc

# ==================================================
# TEST LOOP
# ==================================================

results = []

correct = 0
total = 0

for npc in benchmark["characters"]:

    npc_name = npc["character"]

    print()
    print("===================================")
    print(f"NPC: {npc_name}")
    print("===================================")

    character_info = knowledge.get(
        npc_name.lower(),
        ""
    )

    if character_info == "":
        print(
            f"WARNING: No knowledge found for {npc_name}"
        )

    for test in npc["dialogues"]:

        formatted_prompt = prompt.format(

            race=
                character_info["race"],

            gender=
                character_info["gender"],

            occupation=
                character_info["occupation"],

            home_city=
                character_info["home_city"],

            location=
                character_info["location"],

            morality=
                character_info["morality"],

            aggression=
                character_info["aggression"],

            player_message=
                test["player_message"],

            option_a=
                test["options"]["A"],

            option_b=
                test["options"]["B"],

            option_c=
                test["options"]["C"],

            option_d=
                test["options"]["D"]
        )

        print(formatted_prompt)

        result = chain.invoke({

            # "entity_name":
            #     character_info["entity_name"],

            "race":
                character_info["race"],

            "gender":
                character_info["gender"],

            "occupation":
                character_info["occupation"],

            "home_city":
                character_info["home_city"],

            "location":
                character_info["location"],

            "morality":
                character_info["morality"],

            "aggression":
                character_info["aggression"],

            "player_message":
                test["player_message"],

            "option_a":
                test["options"]["A"],

            "option_b":
                test["options"]["B"],

            "option_c":
                test["options"]["C"],

            "option_d":
                test["options"]["D"]
        })

        prediction = result.content.strip().upper()

        #
        # extract A/B/C/D safely
        #

        match = re.search(
            r"[ABCD]",
            prediction
        )

        if match:
            prediction = match.group(0)
        else:
            prediction = "INVALID"

        expected = test["correct"]

        is_correct = prediction == expected

        if is_correct:
            correct += 1

        total += 1

        results.append({

            "npc": npc_name,

            "step": test["step"],

            "expected": expected,

            "predicted": prediction,

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
# SAVE RESULTS
# ==================================================

pd.DataFrame(results).to_csv(
    "character_knowledge_results.csv",
    index=False
)

print()
print(
    "Saved character_knowledge_results.csv"
)