from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


# modelo para embeddings (rápido y ligero)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# cargar base vectorial
db = FAISS.load_local(
    "vector_db",
    embedding_model,
    allow_dangerous_deserialization=True
)

# modelo LLM
llm = OllamaLLM(model="phi3")


def ask_question(question):

    docs = db.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Responde la pregunta usando el contexto.

Contexto:
{context}

Pregunta:
{question}
"""

    response = llm.invoke(prompt)

    return response