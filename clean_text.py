import json
import re

INPUT_FILE = "datasets/Skyrim_knowledge_new.jsonl"
OUTPUT_FILE = "datasets/Skyrim_knowledge_clean.jsonl"

TARGET_TYPES = {"quest", "location", "character"}

processed = 0
modified_bold = 0
modified_plain = 0
not_found = 0

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in infile:
        processed += 1
        data = json.loads(line)

        if data.get("type") in TARGET_TYPES:
            title = data.get("title", "").strip()
            text = data.get("text", "")

            # 1. Tentar achar o **Título** em negrito
            marker = f"**{title}**"
            index = text.find(marker)

            if index != -1:
                data["text"] = text[index:]
                modified_bold += 1
            else:
                # 2. Tentar achar o Título solto em uma linha (Ex: \nAela the Huntress\n)
                # A Regex procura por uma quebra de linha, o título (ignorando espaços extras), e outra quebra de linha.
                # O \s* lida com espaços fantasmas que o scraping costuma deixar.
                pattern = rf"\n\s*{re.escape(title)}\s*\n"
                
                # Usamos re.finditer para achar TODAS as ocorrências do título solto.
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                if matches:
                    # Se ele achou o título solto, qual deles nós pegamos?
                    # A tabela sempre fica no topo. O último título solto antes de começar um textão longo é o correto.
                    # Pagar a ÚLTIMA ocorrência garante que pulamos o título do cabeçalho do arquivo.
                    last_match = matches[-1]
                    
                    # Corta a partir do começo do título (ignorando o \n inicial que usamos pra ancorar)
                    cut_index = last_match.start() + 1
                    data["text"] = text[cut_index:].strip()
                    modified_plain += 1
                else:
                    not_found += 1

        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

print("====================================")
print("PROCESSAMENTO CONCLUÍDO")
print("====================================")
print(f"Linhas processadas:        {processed}")
print(f"Encontradas com **title**: {modified_bold}")
print(f"Encontradas como linha:    {modified_plain}")
print(f"Não encontradas:           {not_found}")
print(f"Total modificadas:         {modified_bold + modified_plain}")
print(f"Arquivo gerado:            {OUTPUT_FILE}")
print("====================================")