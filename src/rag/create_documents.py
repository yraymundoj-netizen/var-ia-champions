import pandas as pd
from langchain.schema import Document

df = pd.read_csv("data/matches_clean.csv")

documents = []

for _, row in df.iterrows():

    text = f"""
    Season: {row.season}
    Date: {row.date}
    Match: {row.home_team} vs {row.away_team}
    Result: {row.home_goals}-{row.away_goals}
    Stage: {row.Stage}
    """

    documents.append(Document(page_content=text))

print("Documents creados:", len(documents))