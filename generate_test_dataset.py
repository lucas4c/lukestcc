import json
import requests
import random
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ==================================================
# CONFIGURAÇÃO
# ==================================================

# Utilizamos o Llama-3.1-70b-versatile via Groq para geração de dados complexos e raciocínio lógico
generator_model = ChatGroq(
    model="openai/gpt-oss-120b", 
    temperature=0.2 
)

characters_to_process = [
    "Balgruuf the Greater", "Ulfric Stormcloak", "General Tullius", "Delphine",
    "Brynjolf", "Astrid", "Serana", "Isran", "Neloth", "Cicero",
    "Aela the Huntress", "Farkas", "Kodlak Whitemane", "Farengar Secret-Fire",
    "Elisif the Fair", "Maven Black-Briar", "Nazeem", "Paarthurnax",
    "Lydia", "Mercer Frey"
]

def scrape_uesp_dialogue(character_name):
    url_name = character_name.replace(" ", "_")
    url = f"https://en.uesp.net/wiki/Skyrim:{url_name}"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return ""
        
    soup = BeautifulSoup(response.content, 'html.parser')
    
    content = ""
    for paragraph in soup.find_all(['p', 'dl', 'li']):
        text = paragraph.get_text().strip()
        if len(text) > 20: 
            content += text + "\n"
            
    return content[:8000] 

# ==================================================
# PROMPT GERADOR (Corrigido com chaves duplas para LangChain)
# ==================================================

system_prompt = """
You are an expert data curator for a Skyrim dialogue benchmark.
Your task is to take raw text from the UESP Wiki about a specific character and generate a JSON array of 5 dialogue steps.

CRITICAL RULES:
1. The correct answer MUST be an exact quote or highly accurate paraphrase from the game lore provided in the text.
2. The 'player_message' MUST make sense sequentially, creating a 5-step conversation.
3. The FALSE options must follow these difficulty rules to trick baseline LLMs:
   - Make one false option lore-accurate for the world, but wrong for this specific character.
   - Make one false option convey the same information as the correct answer, but in a completely WRONG TONE for the character (e.g., polite instead of aggressive).
   - Make one false option contradict something established in Step 1.

OUTPUT FORMAT MUST BE EXACTLY LIKE THIS (no markdown, just raw JSON).
Note: Always put the correct answer as option A in your JSON output. The python script will randomize the letters later.
[
  {{
    "step": 1,
    "context": [],
    "player_message": "...",
    "options": {{ "A": "[Correct Answer]", "B": "...", "C": "...", "D": "..." }},
    "correct": "A"
  }}
]
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Character: {character}\n\nUESP Raw Text:\n{wiki_text}")
])

chain = prompt_template | generator_model

# ==================================================
# PIPELINE DE GERAÇÃO
# ==================================================

final_dataset = {
    "dataset_name": "skyrim_character_dialogue_benchmark_v2_full",
    "characters": []
}

for char in characters_to_process:
    print(f"[{char}] Extraindo dados...")
    wiki_data = scrape_uesp_dialogue(char)
    
    if not wiki_data:
        print(f"[{char}] Falhou (Sem dados).")
        continue
        
    print(f"[{char}] Gerando árvore via LLM...")
    try:
        result = chain.invoke({
            "character": char,
            "wiki_text": wiki_data
        })
        
        clean_json = result.content.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
            
        dialogues = json.loads(clean_json.strip())
        
        char_block = {
            "character": char,
            "game": "The Elder Scrolls V: Skyrim",
            "conversation_id": f"{char.lower().replace(' ', '_')}_main",
            "dialogues": []
        }
        
        current_context = []
        for step in dialogues:
            options_list = list(step["options"].values())
            correct_text = step["options"]["A"] 
            random.shuffle(options_list)
            
            new_options = {
                "A": options_list[0],
                "B": options_list[1],
                "C": options_list[2],
                "D": options_list[3]
            }
            
            new_correct_letter = [k for k, v in new_options.items() if v == correct_text][0]
            
            processed_step = {
                "step": step["step"],
                "context": list(current_context),
                "player_message": step["player_message"],
                "options": new_options,
                "correct": new_correct_letter
            }
            
            char_block["dialogues"].append(processed_step)
            current_context.append({"speaker": "player", "text": processed_step["player_message"]})
            current_context.append({"speaker": char.lower(), "text": correct_text})

        final_dataset["characters"].append(char_block)
        print(f"[{char}] Concluído!")
        
    except Exception as e:
        print(f"[{char}] Erro: {e}")

with open("datasets/skyrim_benchmark_full.json", "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, indent=2, ensure_ascii=False)