from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import pandas as ps
from vector import retriever_no_guardrails
from vector import retrieve_with_guardrails

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",   # or "llama3-70b-8192"
)

dataset = ps.read_csv('peopleInformation.csv')

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
You are ${targetName}, You are a ${targetGender} ${targetRace} ${targetOccupation},

Here is some relevant information: {information}

${playerName} says: {inquiry}

Respond in character. Your answer MUST be at most 140 characters.,  ${playerName} is ${targetAggression}

Your are ${targetAggression} toward ${playerName}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

playerName = input("WHAT IS YO NAEM?\n")

while True:
    option = input("Who are we talking to? (q to quit)")
    if option == 'q':
        break

    selectedTarget = targetList[int(option) - 1]
    print("\n\n--------------------------------------------------")
    print(f"You are talking to {selectedTarget['targetName']}. (q to leave conversation)")

    while True:

        inquiry = input(f"{playerName}: ")

        if inquiry == 'q':
            break

        information = retrieve_with_guardrails(
            query=inquiry,
            character_name=selectedTarget["targetName"]
        )

        if information is None:
            print(
                f"{selectedTarget['targetName']}: "
                "I don't know enough about that."
            )
            continue

        result = chain.invoke({
            "targetName": selectedTarget['targetName'],
            "targetGender": selectedTarget['targetGender'],
            "targetRace": selectedTarget['targetRace'],
            "targetOccupation": selectedTarget['targetOccupation'],
            "targetAggression": selectedTarget['targetAggression'],
            "information": information,
            "playerName": playerName,
            "inquiry": inquiry
        })

        print(f"{selectedTarget['targetName']}: {result.content}")
