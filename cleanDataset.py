import json
import re

INPUT_FILE = "datasets/Skyrim.jsonl"
OUTPUT_FILE = "datasets/Skyrim_knowledge.jsonl"


# ==================================================
# ENTITY NAME EXTRACTION
# ==================================================

def extract_entity_name(title):

    title = title.strip()

    # Skyrim:Adelaisa Vendicci
    if ":" in title:
        title = title.split(":", 1)[1]

    # Skyrim Adelaisa Vendicci
    if title.startswith("Skyrim "):
        title = title[len("Skyrim "):]

    return title.strip()


# ==================================================
# TYPE DETECTION
# ==================================================

def detect_type(text):

    lower = text.lower()

    #
    # CHARACTER
    #

    if (
        "voice type" in lower and
        "race  |" in lower
    ):
        return "character"

    

    #
    # LOCATION
    #

    if (
        "\nregion" in lower or
        "n# of Zones  |" in lower
    ):
        return "location"
    
    #
    # QUEST
    #

    if (
        "walkthrough" in lower 
    ):
        return "quest"

    #
    # BOOK
    #

    if (
        "book information" in lower
    ):
        return "book"

    return None


# ==================================================
# REMOVE UNWANTED SECTIONS
# ==================================================

def remove_sections(text):

    stop_sections = [
        "## Dialogue",
        "## Conversations",
        "## Generic Dialogue",
        "## Notes",
        "## Bugs",
        "## Appearances",
        "## Gallery",
        "## References",
        "## See Also",
    ]

    for section in stop_sections:

        if section in text:
            text = text.split(section)[0]

    return text


# ==================================================
# REMOVE GAMEPLAY NOISE
# ==================================================

NOISE_PATTERNS = [
    "RefID",
    "BaseID",
    "Health",
    "Magicka",
    "Stamina",
    "Primary Skills",
    "Essential",
    "Follower",
    "StewardHF",
    "Voice Type",
]


def remove_noise(text):

    clean_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        skip = False

        for pattern in NOISE_PATTERNS:

            if pattern.lower() in line.lower():
                skip = True
                break

        if skip:
            continue

        if ".png" in line.lower():
            continue

        if ".jpg" in line.lower():
            continue

        clean_lines.append(line)

    return "\n".join(clean_lines)


# ==================================================
# NORMALIZATION
# ==================================================

def normalize(text):

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


# ==================================================
# CHARACTER / QUEST / LOCATION EXTRACTION
# ==================================================

def extract_intro(text):

    parts = text.split("##")

    intro = parts[0]

    intro = re.sub(r'\n{2,}', '\n', intro)

    return intro.strip()


# ==================================================
# BOOK EXTRACTION
# ==================================================

def extract_book_description(text):

    parts = text.split("##")

    intro = parts[0]

    return normalize(intro)



# ==================================================
# SPLIT BOOKS INTO INDIVIDUAL SENTENCES
# ==================================================

def split_into_sentences(text):

    if not text:
        return []

    text = re.sub(r'\n+', ' ', text)

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 15
    ]


# ==================================================
# PROCESS
# ==================================================

output_documents = []

processed = 0
kept = 0

with open(INPUT_FILE, "r", encoding="utf-8") as infile:

    for raw_line in infile:

        processed += 1

        try:

            page = json.loads(raw_line)

            title = page.get("title", "")
            text = page.get("text", "")

            entity_name = extract_entity_name(title)

            page_type = detect_type(text)

            #
            # Ignore everything else
            #

            if page_type is None:
                continue

            #
            # BOOKS
            #

            if page_type == "book":

                description = extract_book_description(text)

                #
                # Book metadata / description
                #

                if len(description) > 100:

                    output_documents.append({
                        "title": entity_name,
                        "type": "book",
                        "text": description
                    })

                #
                # Book content -> one sentence per document
                #

                content = text

                if content:

                    sentences = split_into_sentences(content)

                    for i, sentence in enumerate(sentences):

                        output_documents.append({
                            "title": entity_name,
                            "type": "book_content",
                            "sentence_id": i,
                            "text": sentence
                        })

                kept += 1
                continue

            #
            # CHARACTER / QUEST / LOCATION
            #

            text = remove_sections(text)
            text = remove_noise(text)

            content = extract_intro(text)
            content = normalize(content)

            if len(content) < 150:
                continue

            output_documents.append({
                "title": entity_name,
                "type": page_type,
                "text": content
            })

            kept += 1

        except Exception as e:

            print(
                f"ERROR processing page #{processed}: {e}"
            )


# ==================================================
# SAVE
# ==================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as outfile:

    for doc in output_documents:

        outfile.write(
            json.dumps(
                doc,
                ensure_ascii=False
            ) + "\n"
        )


# ==================================================
# STATS
# ==================================================

type_counts = {}

for doc in output_documents:

    doc_type = doc["type"]

    type_counts[doc_type] = (
        type_counts.get(doc_type, 0) + 1
    )

print()
print("====================================")
print("PROCESSING COMPLETE")
print("====================================")
print(f"Pages processed: {processed}")
print(f"Pages kept:      {kept}")
print(f"Documents saved: {len(output_documents)}")
print()

for t, count in sorted(type_counts.items()):
    print(f"{t:15} {count}")

print()
print(f"Output file: {OUTPUT_FILE}")
print("====================================")