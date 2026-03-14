import streamlit as st
import re
import unicodedata
import ollama
from typing import Optional

from src.predictor.predict_service import predict_game
from src.predictor.predict_service import df as dataset_df
from src.rag.rag_engine import build_match_context

# CONFIG
st.set_page_config(page_title="Champions AI", page_icon="⚽")

st.title("⚽ Champions AI")

# DATASET
df = dataset_df

if df.empty:
    st.error("❌ No se pudo cargar el dataset")
    st.stop()

# DETECTAR COLUMNAS
home_col = "home_team" if "home_team" in df.columns else "HomeTeam"
away_col = "away_team" if "away_team" in df.columns else "AwayTeam"
home_goals_col = "home_goals" if "home_goals" in df.columns else "HomeGoals"
away_goals_col = "away_goals" if "away_goals" in df.columns else "AwayGoals"


# LISTA EQUIPOS (optimizada)
all_teams = sorted(
    set(df[home_col].unique()).union(df[away_col].unique())
)

# NORMALIZAR TEXTO
def normalize(text: str):

    text = str(text).lower()

    text = unicodedata.normalize("NFKD", text)

    text = text.encode("ascii", "ignore").decode()

    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)

    return re.sub(r"\s+", " ", text).strip()


# PREGUNTA LIBRE
def process_question(question: str) -> str:

    dataset_ans = check_dataset(question)

    if dataset_ans:
        return dataset_ans

    context_matches = build_match_context(df, question)

    # si RAG no encuentra nada
    if not context_matches or not context_matches.strip():
        return "📊 No encontré datos relevantes en el dataset."

    context = f"""
Base de datos Champions League

Total partidos: {len(df)}

Datos relevantes del dataset:
{context_matches}
"""

    try:

        response = ollama.chat(
            model="phi3",
            messages=[
                {
                    "role": "system",
                    "content": f"""
Eres un analista experto en Champions League.

Usa SOLO los datos del contexto.

Reglas:
- No inventes información
- No uses conocimiento externo
- Si no hay datos suficientes di:
"No hay datos suficientes en el dataset"

Responde máximo en 2 frases usando emojis ⚽📊🏆.

Contexto:
{context}
"""
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        return response["message"]["content"].strip()

    except Exception as e:

        print("Ollama error:", e)

        return "🤖 No pude responder en este momento."


# RESPUESTA DESDE DATASET
def check_dataset(question: str) -> Optional[str]:

    q_norm = normalize(question)

    year_match = re.search(r"(19|20)\d{2}", q_norm)

    if year_match:

        year = year_match.group()

        if "season" in df.columns:

            season = df[df["season"].astype(str).str.contains(year)]

            if not season.empty:

                final = season.iloc[-1]

                home = final[home_col]
                away = final[away_col]

                hg = int(final[home_goals_col])
                ag = int(final[away_goals_col])

                winner = home if hg > ag else away

                return f"""
🏆 **Champions League {year}**

⚽ {home} {hg}-{ag} {away}

🥇 **Campeón: {winner}**
"""

    return None


# PREDICCIÓN
def predict_match(home_team: str, away_team: str):

    try:
        return predict_game(home_team, away_team)

    except Exception as e:

        st.error(f"❌ Error: {e}")

        return None


# UI
tab1, tab2 = st.tabs(["❓ Pregunta libre", "⚽ Predictor Equipos"])


# TAB 1 - CHAT

with tab1:

    st.markdown("### 📝 Haz cualquier pregunta")

    question = st.text_input(
        "Ej: ¿Quién ganó la Champions 2023?"
    )

    if question:

        with st.spinner("🤖 Pensando..."):

            answer = process_question(question)

            st.markdown(answer)


# TAB 2 - PREDICTOR

with tab2:

    st.markdown("### ⚽ Selecciona los equipos")

    col1, col2 = st.columns(2)

    with col1:

        home_team = st.selectbox(
            "🏠 Equipo Local",
            options=all_teams
        )

    with col2:

        away_options = [t for t in all_teams if t != home_team]

        away_team = st.selectbox(
            "✈️ Equipo Visitante",
            options=away_options
        )

    if st.button("🔮 **PREDICIR**", type="primary"):

        with st.spinner("🔮 Calculando probabilidades..."):

            prediction = predict_match(home_team, away_team)

            if prediction:

                st.success(
                    f"⚽ **{prediction['home_team']} vs {prediction['away_team']}**"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "🏠 Local",
                        f"{prediction['home_win_pct']:.0f}%"
                    )

                with col2:
                    st.metric(
                        "🤝 Empate",
                        f"{prediction['draw_pct']:.0f}%"
                    )

                with col3:
                    st.metric(
                        "✈️ Visitante",
                        f"{prediction['away_win_pct']:.0f}%"
                    )

                st.info(
                    f"🎯 **Marcador más probable:** {prediction['most_likely_score']}"
                )


# EQUIPOS DISPONIBLES

with st.expander("📋 Equipos disponibles"):

    st.write(f"**{len(all_teams)} equipos**")

    st.write(", ".join(all_teams[:20]))

    if len(all_teams) > 20:
        st.write("... y más")