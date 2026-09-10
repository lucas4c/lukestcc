import os
import json
import re
import pandas as ps
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

MEMORY_DIR = "memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

DEBUG = False

def safe_filename(name):
    """Sanitize NPC name so it becomes a valid safe filename."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)

def load_memory(npc_name):
    file_path = os.path.join(MEMORY_DIR, f"{safe_filename(npc_name)}.json")

    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if DEBUG:
                print(f"[DEBUG] Loaded memory from {file_path}")
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load memory: {e}")
        return []


def save_memory(npc_name, memory_list):
    file_path = os.path.join(MEMORY_DIR, f"{safe_filename(npc_name)}.json")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(memory_list, f, indent=2, ensure_ascii=False)
        if DEBUG:
            print(f"[DEBUG] Memory saved to {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save memory: {e}")


def add_memory(npc_name, player_name, player_msg, npc_msg):
    memory_list = load_memory(npc_name)

    memory_list.append({
        "player": player_name,
        "player_message": player_msg,
        "npc_message": npc_msg
    })

    if DEBUG:
        print("[DEBUG] Adding new memory entry...")

    save_memory(npc_name, memory_list)


def format_memory(memory_list):
    """Format memory (last 10 only) for prompt context."""
    if not memory_list:
        return "No past interactions."

    lines = []
    for entry in memory_list[-10:]:
        lines.append(f"Player said: {entry['player_message']} | NPC replied: {entry['npc_message']}")

    return "\n".join(lines)

model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

dataset = ps.read_csv("peopleInformation.csv")

targetList = []
for _, row in dataset.iterrows():
    targetList.append({
        "targetName": row["name"],
        "targetGender": row["gender"],
        "targetRace": row["race"],
        "targetOccupation": row["class"],
        "targetAggression": row["aggression_level"]
    })

template = """
You are ${targetName}, a ${targetGender} ${targetRace} ${targetOccupation}.

Your mood toward ${playerName} is: ${targetAggression}.

Here is your memory of past interactions:
{memory}

Here is relevant world information:
{information}

${playerName} says: {inquiry}

Respond IN CHARACTER.
Your answer MUST be 140 characters or less.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

playerName = input("What is your name? ")

while True:
    print("\nChoose an NPC to talk to:")
    for i, t in enumerate(targetList, start=1):
        print(f"{i}. {t['targetName']}")

    option = input("(q to quit)\n")

    if option.lower() == "q":
        break

    try:
        selectedTarget = targetList[int(option) - 1]
    except:
        print("Invalid option.")
        continue

    npc = selectedTarget["targetName"]
    print("\n--------------------------------------------------")
    print(f"You are now talking to {npc}. (type q to exit chat)\n")

    while True:
        inquiry = input(f"{playerName}: ")

        if inquiry.lower() == "q":
            break

        information = retriever.invoke(inquiry)

        old_memory = load_memory(npc)

        result = chain.invoke({
            "targetName": selectedTarget["targetName"],
            "targetGender": selectedTarget["targetGender"],
            "targetRace": selectedTarget["targetRace"],
            "targetOccupation": selectedTarget["targetOccupation"],
            "targetAggression": selectedTarget["targetAggression"],
            "information": information,
            "playerName": playerName,
            "inquiry": inquiry,
            "memory": format_memory(old_memory)
        })

        npc_response = result.content
        print(f"{npc}: {npc_response}")

        add_memory(npc, playerName, inquiry, npc_response)

        updated_memory = load_memory(npc)

        # print("\n[Updated Memory]")
        # print(format_memory(updated_memory))
        # print("--------------------------------------------------")
