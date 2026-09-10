from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_json(
        "datasets/Skyrim_knowledge.jsonl",
        lines=True
    )

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row, in df.iterrows():


        document = Document(
            page_content = str(row["text"]),
            metadata ={
                "entity": row["title"],
                "type": row["type"]
            },
            id=str(i)
        )

        ids.append(str(i))
        documents.append(document)


    

vector_store = Chroma(
    collection_name = "people_of_the_world",
    persist_directory = db_location,
    embedding_function = embeddings
)

if add_documents:

    batch_size = 100

    total_docs = len(documents)

    for start in range(0, total_docs, batch_size):

        end = min(start + batch_size, total_docs)

        batch_documents = documents[start:end]
        batch_ids = ids[start:end]

        vector_store.add_documents(
            documents=batch_documents,
            ids=batch_ids
        )

        print(
            f"Added {end}/{total_docs} documents "
            f"({100 * end / total_docs:.2f}%)"
        )


retriever_no_guardrails = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

def retrieve_with_guardrails(
    query,
    character_name=None,
    k=5,
    score_threshold=0.45,
    min_context_chars=40
):

    results = vector_store.similarity_search_with_score(
        query,
        k=k,
        # filter={
        #     "character": character_name
        # } if character_name else None
    )

    filtered_docs = []

    print("\n========== RETRIEVAL ==========")

    for doc, score in results:

        print(f"SCORE: {score}")
        print(f"CONTENT: {doc.page_content}")
        print("--------------------------------")

        # LOWER SCORE = BETTER MATCH
        if score < score_threshold:
            filtered_docs.append(doc)

    if len(filtered_docs) == 0:
        print("GUARDRAIL BLOCKED: no relevant documents")
        return None

    combined_context = "\n".join(
        [doc.page_content for doc in filtered_docs]
    )

    # SECOND THRESHOLD:
    # minimum amount of semantic context
    if len(combined_context) < min_context_chars:
        print("GUARDRAIL BLOCKED: insufficient context")
        return None

    print("GUARDRAIL PASSED")

    return combined_context
