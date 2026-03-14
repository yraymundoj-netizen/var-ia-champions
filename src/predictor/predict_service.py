import os
import pandas as pd
from rapidfuzz import process, fuzz
from typing import Dict, Any
import re
import warnings

warnings.filterwarnings("ignore")

# IMPORTAR POISSON
try:
    from src.predictor.poisson_predictor import PoissonPredictor
    POISSON_AVAILABLE = True
    print("✅ PoissonPredictor disponible")
except Exception as e:
    print(f"⚠️ PoissonPredictor no disponible: {e}")
    POISSON_AVAILABLE = False


# CARGAR DATASET
def load_champions_data():

    try:

        path = os.path.join("data", "champ_clean.csv")

        print(f"📂 Cargando dataset: {path}")

        df = pd.read_csv(path)

        rename_map = {
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "HomeGoals": "home_goals",
            "AwayGoals": "away_goals",
            "Season": "season"
        }

        df = df.rename(columns=rename_map)

        required = ["home_team", "away_team", "home_goals", "away_goals"]

        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")

        print(f"✅ Dataset cargado: {len(df)} partidos")

        return df

    except Exception as e:

        print("❌ Error cargando dataset:", e)

        return pd.DataFrame()


# DATASET GLOBAL
df = load_champions_data()

# lista de equipos
teams_dataset = sorted(
    list(set(df["home_team"].tolist() + df["away_team"].tolist()))
)


# INICIALIZAR POISSON
predictor = None

if POISSON_AVAILABLE and not df.empty:

    try:

        predictor = PoissonPredictor(df)

        print("✅ Modelo Poisson inicializado")

    except Exception as e:

        print(f"⚠️ Error inicializando Poisson: {e}")


# NORMALIZAR EQUIPO
def normalize_team(text: str) -> str:

    text = str(text).lower().strip()

    text = re.sub(r"[^a-záéíóúñ\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text


# MATCH DE EQUIPOS
def get_real_team_name(name: str) -> str:

    name_norm = normalize_team(name)

    teams_norm = [normalize_team(t) for t in teams_dataset]

    match = process.extractOne(name_norm, teams_norm, scorer=fuzz.partial_ratio)

    if match and match[1] > 70:

        return teams_dataset[match[2]]

    return name


# PREDICCIÓN POISSON
def safe_predict_match(home_team: str, away_team: str) -> Dict[str, Any]:

    home_team = get_real_team_name(home_team)
    away_team = get_real_team_name(away_team)

    print(f"🔮 Prediciendo: {home_team} vs {away_team}")

    if predictor:

        try:

            raw_result = predictor.predict_match(home_team, away_team)

            print("✅ Poisson OK")

            return standardize_prediction(raw_result)

        except Exception as e:

            print(f"⚠️ Poisson falló: {e}")

    return statistical_fallback(home_team, away_team)


# ESTANDARIZAR RESULTADO
def standardize_prediction(raw_result: Dict) -> Dict[str, Any]:

    home_win = raw_result.get("home_win_pct", 33)
    away_win = raw_result.get("away_win_pct", 33)
    draw = raw_result.get("draw_pct", 34)

    score = raw_result.get("most_likely_score", "1-1")

    return {

        "home_team": raw_result.get("home_team"),

        "away_team": raw_result.get("away_team"),

        "home_win_pct": round(home_win, 1),

        "away_win_pct": round(away_win, 1),

        "draw_pct": round(draw, 1),

        "most_likely_score": score
    }


# FALLBACK ESTADÍSTICO
def statistical_fallback(home_team: str, away_team: str) -> Dict[str, Any]:

    print("📊 Usando fallback estadístico")

    avg_home_goals = df["home_goals"].mean()
    avg_away_goals = df["away_goals"].mean()

    home_games = df[df["home_team"] == home_team]
    away_games = df[df["away_team"] == away_team]

    home_factor = home_games["home_goals"].mean() / avg_home_goals if not home_games.empty else 1
    away_factor = away_games["away_goals"].mean() / avg_away_goals if not away_games.empty else 1

    home_win_pct = 35 * home_factor
    away_win_pct = 30 * away_factor
    draw_pct = 35

    total = home_win_pct + away_win_pct + draw_pct

    home_win_pct = (home_win_pct / total) * 100
    away_win_pct = (away_win_pct / total) * 100
    draw_pct = (draw_pct / total) * 100

    score = f"{int(avg_home_goals)}-{int(avg_away_goals)}"

    return {

        "home_team": home_team,

        "away_team": away_team,

        "home_win_pct": round(home_win_pct, 1),

        "away_win_pct": round(away_win_pct, 1),

        "draw_pct": round(draw_pct, 1),

        "most_likely_score": score
    }


# FUNCIÓN PRINCIPAL
def predict_game(home_team: str, away_team: str) -> Dict[str, Any]:

    try:

        result = safe_predict_match(home_team, away_team)

        required = [
            "home_team",
            "away_team",
            "home_win_pct",
            "away_win_pct",
            "draw_pct",
            "most_likely_score"
        ]

        for key in required:

            if key not in result:
                raise ValueError(f"Clave faltante: {key}")

        total = result["home_win_pct"] + result["away_win_pct"] + result["draw_pct"]

        if not 95 <= total <= 105:
            print(f"⚠️ Ajustando porcentajes: {total:.1f}%")

        print(f"✅ Predicción OK: {result['most_likely_score']}")

        return result

    except Exception as e:

        print(f"❌ ERROR FINAL: {e}")

        return {

            "home_team": home_team,
            "away_team": away_team,
            "home_win_pct": 33.3,
            "away_win_pct": 33.3,
            "draw_pct": 33.3,
            "most_likely_score": "1-1"
        }


# TEST
if __name__ == "__main__":

    print("🧪 Test predict_game")

    result = predict_game("Real Madrid", "Barcelona")

    print(result)