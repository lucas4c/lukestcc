import json
import re

INPUT_FILE = "datasets/Skyrim_knowledge.jsonl"
OUTPUT_FILE = "datasets/Skyrim_characters_structured.jsonl"


def extract_field(pattern, text):

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).strip()

    return "Unknown"


def extract_character_info(doc):

    text = doc["text"]

    return {

        "entity_name":
            doc["title"],

        "entity_type":
            "character",

        "race":
            extract_field(
                r"Race\s*\|\s*(.+?)\s*\|",
                text
            ),

        "gender":
            extract_field(
                r"Gender\s*\|\s*([^\n]+)",
                text
            ),

        "occupation":
            extract_field(
                r"Class\s*\|\s*([^\n]+)",
                text
            ),

        "home_city":
            extract_field(
                r"Home City\s*\|\s*([^\n]+)",
                text
            ),

        "location":
            extract_field(
                r"Location\s*\|\s*([^\n]+)",
                text
            ),

        "store":
            extract_field(
                r"Store\s*\|\s*([^\n]+)",
                text
            ),

        "morality":
            extract_field(
                r"Morality\s*\|\s*(.+?)\s*\|",
                text
            ),

        "aggression":
            extract_field(
                r"Aggression\s*\|\s*([^\n]+)",
                text
            )
    }


output = []

processed = 0
saved = 0

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as infile:

    for line in infile:

        processed += 1

        try:

            doc = json.loads(line)

            if doc.get("type") != "character":
                continue

            character = extract_character_info(doc)

            output.append(character)

            saved += 1

        except Exception as e:

            print(
                f"Error processing line {processed}: {e}"
            )


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as outfile:

    for row in output:

        outfile.write(
            json.dumps(
                row,
                ensure_ascii=False
            ) + "\n"
        )


print()
print("===================================")
print("CHARACTER EXTRACTION COMPLETE")
print("===================================")
print(f"Processed: {processed}")
print(f"Characters: {saved}")
print(f"Output: {OUTPUT_FILE}")
print("===================================")