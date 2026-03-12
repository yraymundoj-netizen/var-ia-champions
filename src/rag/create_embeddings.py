import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_documents():

    df = pd.read_csv("data/matches_clean.csv")

    # limpiar posibles valores vacíos
    df = df.dropna()

    docs = []

    for _, row in df.iterrows():

        home = str(row["home_team"])
        away = str(row["away_team"])
        season = str(row["season"])
        round_ = str(row["Round"])
        home_goals = str(row["home_goals"])
        away_goals = str(row["away_goals"])

        text = (
            f"UEFA Champions League match. "
            f"Season {season}, Round {round_}. "
            f"{home} vs {away}. "
            f"Final score: {home} {home_goals} - {away_goals} {away}. "
            f"Teams: {home}, {away}. "
            f"Competition: Champions League."
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "season": season,
                    "round": round_,
                    "home_team": home,
                    "away_team": away,
                    "match": f"{home} vs {away}"
                }
            )
        )

    return docs


def create_vector_store():

    print("Loading documents...")
    documents = load_documents()

    print(f"Total documents loaded: {len(documents)}")

    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    print("Saving vector database...")
    vectorstore.save_local("vector_db")

    print("Vector database created successfully!")


if __name__ == "__main__":
    create_vector_store()