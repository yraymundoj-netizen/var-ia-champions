import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_documents():
    df = pd.read_csv("data/matches-demo.csv")

    docs = []

    for row in df["text"]:
        docs.append(Document(page_content=row))

    return docs


def create_vector_store():

    print("Loading documents...")
    documents = load_documents()

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    print("Saving vector database...")
    vectorstore.save_local("vector_db")

    print("Done!")


if __name__ == "__main__":
    create_vector_store()