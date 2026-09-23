import os
import re
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# ==================================================
# 1. FUNÇÃO DE LIMPEZA HEURÍSTICA (Data Sanitization)
# ==================================================
def sanitize_wiki_text(text):
    """
    Limpa o texto cru da UESP Wiki antes de vetorizar.
    Remove tabelas de status, links de navegação e avisos irrelevantes para a lore.
    """
    # Remove tabelas (linhas que começam com | ou ---)
    text = re.sub(r'^\s*\|.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*---.*$', '', text, flags=re.MULTILINE)
    
    # Remove avisos genéricos de edição da Wiki
    text = text.replace("**This page or section is incomplete. You can help by adding to it.**", "")
    text = text.replace("For more information, see the help files, the style guide, and this article's talk page.", "")
    text = text.replace("(view on map) (lore page)", "")
    text = text.replace("(view on map)", "")
    
    # Remove rodapés de navegação como "Prev. | Volume VI | Next"
    text = re.sub(r'\bPrev\.\s*\|.*?\|\s*Next\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bUp\s*\|.*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove quebras de linha múltiplas geradas pela limpeza
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

# ==================================================
# 2. CONFIGURAÇÃO BASE (Splitter e Embeddings)
# ==================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""]
)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chroma_langchain_db"

# Lê os dados brutos
df = pd.read_json("datasets/Skyrim_knowledge.jsonl", lines=True)

# ==================================================
# 3. GERAÇÃO DE DOCUMENTOS (Chunks limpos)
# ==================================================
add_documents = not os.path.exists(db_location)
documents = []

if add_documents:
    print("🧹 Sanitizando e fatiando dados da Wiki...")
    chunk_id_counter = 0

    for i, row in df.iterrows():
        raw_text = str(row["text"])
        clean_text = sanitize_wiki_text(raw_text)
        
        # Só fatia e salva se sobrar texto útil após a limpeza
        if len(clean_text) < 20: 
            continue
            
        chunks = text_splitter.split_text(clean_text)
        
        for chunk in chunks:
            # Metadata enriquecido para filtros futuros
            document = Document(
                page_content=chunk,
                metadata={
                    "entity": row.get("title", "Unknown"),
                    "type": row.get("type", "lore"),
                    "source_id": str(i)
                },
                id=str(chunk_id_counter)
            )
            documents.append(document)
            chunk_id_counter += 1

# ==================================================
# 4. CRIAÇÃO DO BANCO (ChromaDB)
# ==================================================
vector_store = Chroma(
    collection_name="people_of_the_world",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents and documents:
    print(f"📦 Indexando {len(documents)} chunks no ChromaDB...")
    batch_size = 100
    total_docs = len(documents)
    ids = [doc.id for doc in documents]

    for start in range(0, total_docs, batch_size):
        end = min(start + batch_size, total_docs)
        vector_store.add_documents(
            documents=documents[start:end],
            ids=ids[start:end]
        )
        print(f"   Progresso: {end}/{total_docs} ({100 * end / total_docs:.1f}%)")
    print("✅ Banco Vetorial criado com sucesso!")
elif not add_documents:
    # Se o banco já existe, precisamos carregar os documentos para alimentar o BM25
    # O Chroma não devolve todos os docs facilmente, então lemos do dataset original de novo
    print("🔄 Banco Chroma detectado. Carregando documentos base para o BM25...")
    chunk_id_counter = 0
    for i, row in df.iterrows():
        raw_text = str(row["text"])
        clean_text = sanitize_wiki_text(raw_text)
        if len(clean_text) < 20: continue
        chunks = text_splitter.split_text(clean_text)
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={"entity": row.get("title", "Unknown"), "type": row.get("type", "lore"), "source_id": str(i)}
            ))

# ==================================================
# 5. CONFIGURAÇÃO DA BUSCA HÍBRIDA (Ensemble Retriever)
# ==================================================
# Retriever 1: BM25 (Busca Léxica/Palavras-chave exatas)
# Excelente para nomes próprios bizarros de Skyrim (Fahlbtharz, Paarthurnax)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

# Retriever 2: Chroma (Busca Semântica via mxbai)
# Excelente para entender contexto e sinônimos (War = Conflict)
chroma_retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# O Ensemble junta os resultados dos dois, remove duplicatas e faz o re-ranking (peso igual 50/50)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5]
)

# ==================================================
# 6. FUNÇÃO RAG EXPORTÁVEL (Com Guardrails e Filtros)
# ==================================================
def retrieve_with_guardrails(
    query,
    filter_type=None, 
    k_final=4,
    min_context_chars=40
):
    """
    Busca Híbrida (Semântica + Léxica) com re-ranking automático.
    Opcional: Filtrar por tipo (ex: filter_type="location")
    """
    # Se um filtro for passado, aplicamos a restrição diretamente na busca vetorial do Chroma
    # O BM25 não suporta filtros de metadados nativamente da mesma forma, mas o re-ranking final compensará.
    if filter_type:
        chroma_retriever.search_kwargs["filter"] = {"type": filter_type}
    else:
        # Reseta o filtro se não for passado
        chroma_retriever.search_kwargs.pop("filter", None)

    print("\n========== HYBRID RETRIEVAL (BM25 + Chroma) ==========")
    
    # O invoke do Ensemble já roda os 2 bancos e junta os melhores
    results = ensemble_retriever.invoke(query)
    
    # Cortamos para o limite k_final (pois 3 do BM25 + 3 do Chroma podem dar até 6 resultados)
    top_docs = results[:k_final]

    filtered_docs = []
    
    for i, doc in enumerate(top_docs):
        entity = doc.metadata.get('entity', 'Unknown')
        print(f"Rank {i+1} | ENTITY: {entity}")
        print(f"CONTENT: {doc.page_content.strip()[:120]}...")
        print("--------------------------------")
        filtered_docs.append(doc)

    if not filtered_docs:
        print("GUARDRAIL BLOCKED: No documents found.")
        return None

    combined_context = "\n".join([doc.page_content.strip() for doc in filtered_docs])

    # Threshold de tamanho de contexto
    if len(combined_context) < min_context_chars:
        print("GUARDRAIL BLOCKED: Insufficient context size.")
        return None

    print(f"GUARDRAIL PASSED: {len(filtered_docs)} chunks approved.")
    return combined_context

# O objeto a ser importado pelo test3_rag.py agora é a função híbrida (ou o ensemble direto)
retriever = ensemble_retriever