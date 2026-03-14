import re
from rapidfuzz import process, fuzz
from typing import Optional, List, Dict, Any
import unicodedata

from src.predictor.predict_service import predict_game
from src.predictor.predict_service import df as dataset_df
from src.predictor.predict_service import get_real_team_name


# -------------------------
# DATASET (MISMO QUE POISSON)
# -------------------------

df = dataset_df

teams_list = sorted(list(set(df["HomeTeam"]).union(set(df["AwayTeam"]))))


# -------------------------
# NORMALIZAR NOMBRES
# -------------------------

def normalize_team_name(text: str) -> str:

    text = str(text).lower().strip()

    if "›" in text:
        text = text.split("›")[0].strip()

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")

    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------
# DETECTAR EQUIPOS
# -------------------------

def extract_teams(question: str) -> List[str]:

    question_norm = normalize_team_name(question)

    teams_norm = [normalize_team_name(t) for t in teams_list]

    matches = process.extract(
        question_norm,
        teams_norm,
        scorer=fuzz.partial_ratio,
        limit=8,
        score_cutoff=70
    )

    detected = []

    for team_norm, score, idx in matches:

        if score >= 75 and len(detected) < 2:

            detected.append(teams_list[idx])

    return detected


# -------------------------
# FUNCIÓN PRINCIPAL RAG
# -------------------------

def ask_question(question: str) -> Optional[Dict[str, Any]]:

    teams = extract_teams(question)

    if len(teams) < 2:
        return None

    teamA, teamB = teams

    try:

        prediction = predict_game(teamA, teamB)

        rag_response = {

            "teams_detected": [
                prediction["home_team"],
                prediction["away_team"]
            ],

            "prediction": format_rag_prediction(prediction),

            "stats": get_h2h_stats(teamA, teamB),

            "confidence": "high"

        }

        return rag_response

    except Exception as e:

        print(f"❌ Error en predicción RAG: {e}")

        return None


# -------------------------
# FORMATO RESPUESTA
# -------------------------

def format_rag_prediction(prediction: Dict[str, Any]) -> str:

    home = prediction["home_team"]
    away = prediction["away_team"]

    response = f"""
🔥 **PREDICCIÓN: {home} vs {away}**

📊 **Probabilidades**
• {home} gana → **{prediction['home_win_pct']:.1f}%**
• Empate → **{prediction['draw_pct']:.1f}%**
• {away} gana → **{prediction['away_win_pct']:.1f}%**

🎯 **Marcador probable:** {prediction['most_likely_score']}
"""

    return response.strip()


# -------------------------
# H2H HISTÓRICO
# -------------------------

def get_h2h_stats(teamA: str, teamB: str) -> str:

    realA = get_real_team_name(teamA)
    realB = get_real_team_name(teamB)

    h2h_home = df[(df["HomeTeam"] == realA) & (df["AwayTeam"] == realB)]

    h2h_away = df[(df["HomeTeam"] == realB) & (df["AwayTeam"] == realA)]

    total = len(h2h_home) + len(h2h_away)

    if total == 0:
        return ""

    winsA = (
        len(h2h_home[h2h_home["HomeGoals"] > h2h_home["AwayGoals"]])
        +
        len(h2h_away[h2h_away["AwayGoals"] > h2h_away["HomeGoals"]])
    )

    winsB = (
        len(h2h_home[h2h_home["HomeGoals"] < h2h_home["AwayGoals"]])
        +
        len(h2h_away[h2h_away["HomeGoals"] > h2h_away["AwayGoals"]])
    )

    draws = total - winsA - winsB

    return f"""
📈 **Historial ({total} partidos)**

• {realA} victorias → {winsA}
• {realB} victorias → {winsB}
• Empates → {draws}
"""


# -------------------------
# DEBUG
# -------------------------

def debug_teams(question: str):

    teams = extract_teams(question)

    print("Pregunta:", question)

    print("Equipos detectados:", teams)

    return teams


# -------------------------
# TEST
# -------------------------

if __name__ == "__main__":

    tests = [
        "Real Madrid vs Barcelona",
        "Bayern Munich vs Manchester City",
        "PSG contra Chelsea"
    ]

    for q in tests:

        result = ask_question(q)

        print("\n", q)

        if result:
            print(result["prediction"])
        else:
            print("No se detectaron equipos")