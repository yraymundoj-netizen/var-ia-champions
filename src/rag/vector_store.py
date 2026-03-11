from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from create_embeddings import load_documents


def create_vector_db():

    docs = load_documents()

    embeddings = OllamaEmbeddings(model="phi3")

    db = FAISS.from_documents(docs, embeddings)

    db.save_local("vector_db")

    print("Vector DB creada")