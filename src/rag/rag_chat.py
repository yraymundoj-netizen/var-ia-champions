from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import pandas as pd

# modelo embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# cargar vector DB
db = FAISS.load_local(
    "vector_db",
    embedding_model,
    allow_dangerous_deserialization=True
)

# modelo LLM
llm = OllamaLLM(model="phi3")

# cargar dataset original
df = pd.read_csv("data/matches_clean.csv")


def search_matches(question):

    teams = df["home_team"].unique()

    found_teams = []

    for team in teams:
        if team.lower() in question.lower():
            found_teams.append(team)

    if len(found_teams) >= 2:

        team1 = found_teams[0]
        team2 = found_teams[1]

        matches = df[
            ((df["home_team"] == team1) & (df["away_team"] == team2)) |
            ((df["home_team"] == team2) & (df["away_team"] == team1))
        ]

        results = []

        for _, row in matches.iterrows():

            text = f"{row['home_team']} {row['home_goals']} - {row['away_goals']} {row['away_team']} (Season {row['season']})"

            results.append(text)

        return "\n".join(results)

    return None


def ask_question(question):

    structured_results = search_matches(question)

    if structured_results:
        context = structured_results
    else:

        docs = db.similarity_search(question, k=5)

        context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Responde usando SOLO la información del contexto.

No inventes información.
No agregues datos que no aparezcan en el contexto.

Si no encuentras la respuesta en el contexto responde:
"No se encontró información en la base de datos."

Contexto:
{context}

Pregunta:
{question}

Respuesta clara:
"""

    response = llm.invoke(prompt)

    return response