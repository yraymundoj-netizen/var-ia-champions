import re
import os
import numpy as np
import pandas as pd
from scipy.stats import poisson
from typing import Optional


# ─────────────────────────────────────────────────────────────
# RUTA DEL PROYECTO
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(BASE_DIR, "data", "champ_clean.csv")


# ─────────────────────────────────────────────────────────────
# LIMPIEZA DE NOMBRE DE EQUIPO
# ─────────────────────────────────────────────────────────────

def clean_team_name(raw: str) -> str:
    name = re.split(r"\s*›", str(raw))[0]
    return name.strip()


# ─────────────────────────────────────────────────────────────
# CARGA DE DATASET
# ─────────────────────────────────────────────────────────────

def load_champions_data(
    filepath: str = CSV_PATH,
    season_from: Optional[str] = None,
    season_to: Optional[str] = None,
    stage: Optional[str] = None,
) -> pd.DataFrame:

    print(f"📂 Cargando dataset: {filepath}")

    df = pd.read_csv(filepath)

    # NORMALIZAR COLUMNAS
    rename_map = {
        "home_team": "HomeTeam",
        "away_team": "AwayTeam",
        "home_goals": "HomeGoals",
        "away_goals": "AwayGoals",
        "Season": "season"
    }

    df.rename(columns=rename_map, inplace=True)

    required = {"HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Columnas faltantes en CSV: {missing}")

    # limpiar nombres
    df["HomeTeam"] = df["HomeTeam"].apply(clean_team_name)
    df["AwayTeam"] = df["AwayTeam"].apply(clean_team_name)

    # convertir goles
    df["HomeGoals"] = pd.to_numeric(df["HomeGoals"], errors="coerce")
    df["AwayGoals"] = pd.to_numeric(df["AwayGoals"], errors="coerce")

    df.dropna(subset=["HomeGoals", "AwayGoals"], inplace=True)

    df["HomeGoals"] = df["HomeGoals"].astype(int)
    df["AwayGoals"] = df["AwayGoals"].astype(int)

    # filtros
    if season_from and "season" in df.columns:
        df = df[df["season"] >= season_from]

    if season_to and "season" in df.columns:
        df = df[df["season"] <= season_to]

    if stage and "Stage" in df.columns:
        df = df[df["Stage"].str.contains(stage, case=False, na=False)]

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No hay partidos tras aplicar filtros.")

    print(f"✅ Dataset listo: {len(df)} partidos")

    return df


# ─────────────────────────────────────────────────────────────
# MODELO POISSON + DIXON COLES
# ─────────────────────────────────────────────────────────────

class PoissonPredictor:

    MAX_GOALS = 10

    # factor Dixon-Coles
    RHO = -0.1

    def __init__(self, data: pd.DataFrame) -> None:

        self._validate_input(data)

        self.data = data.copy()

        self.attack_strength = {}
        self.defense_strength = {}

        self.avg_home_goals = 0
        self.avg_away_goals = 0

        self.teams = []

        self._train()

    # ─────────────────────────
    # VALIDACIÓN
    # ─────────────────────────

    def _validate_input(self, data: pd.DataFrame):

        rename_map = {
            "home_team": "HomeTeam",
            "away_team": "AwayTeam",
            "home_goals": "HomeGoals",
            "away_goals": "AwayGoals",
        }

        data.rename(columns=rename_map, inplace=True)

        required = {"HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"}

        missing = required - set(data.columns)

        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")

        if data.empty:
            raise ValueError("Dataset vacío")

    # ─────────────────────────
    # ENTRENAMIENTO
    # ─────────────────────────

    def _train(self):

        df = self.data

        self.avg_home_goals = df["HomeGoals"].mean()
        self.avg_away_goals = df["AwayGoals"].mean()

        global_avg = (self.avg_home_goals + self.avg_away_goals) / 2

        self.teams = sorted(
            set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
        )

        for team in self.teams:

            home_mask = df["HomeTeam"] == team
            away_mask = df["AwayTeam"] == team

            goals_scored = np.concatenate([
                df.loc[home_mask, "HomeGoals"].values,
                df.loc[away_mask, "AwayGoals"].values
            ])

            goals_conceded = np.concatenate([
                df.loc[home_mask, "AwayGoals"].values,
                df.loc[away_mask, "HomeGoals"].values
            ])

            avg_scored = goals_scored.mean() if len(goals_scored) else global_avg
            avg_conceded = goals_conceded.mean() if len(goals_conceded) else global_avg

            self.attack_strength[team] = avg_scored / global_avg
            self.defense_strength[team] = avg_conceded / global_avg

    # ─────────────────────────
    # GOLES ESPERADOS
    # ─────────────────────────

    def _expected_goals(self, home_team, away_team):

        lh = (
            self.attack_strength[home_team]
            * self.defense_strength[away_team]
            * self.avg_home_goals
        )

        la = (
            self.attack_strength[away_team]
            * self.defense_strength[home_team]
            * self.avg_away_goals
        )

        return lh, la

    # ─────────────────────────
    # MATRIZ POISSON + DIXON COLES
    # ─────────────────────────

    def _score_matrix(self, lh, la):

        goals = np.arange(0, self.MAX_GOALS + 1)

        matrix = np.outer(
            poisson.pmf(goals, lh),
            poisson.pmf(goals, la)
        )

        # Dixon-Coles correction
        matrix[0, 0] *= (1 - self.RHO)
        matrix[1, 1] *= (1 - self.RHO)
        matrix[0, 1] *= (1 + self.RHO)
        matrix[1, 0] *= (1 + self.RHO)

        matrix /= matrix.sum()

        return matrix

    # ─────────────────────────
    # PREDICCIÓN
    # ─────────────────────────

    def predict_match(self, home_team, away_team):

        if home_team not in self.teams:
            raise ValueError(f"Equipo no encontrado: {home_team}")

        if away_team not in self.teams:
            raise ValueError(f"Equipo no encontrado: {away_team}")

        lh, la = self._expected_goals(home_team, away_team)

        matrix = self._score_matrix(lh, la)

        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()

        most_likely = np.unravel_index(np.argmax(matrix), matrix.shape)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_pct": round(home_win * 100, 2),
            "draw_pct": round(draw * 100, 2),
            "away_win_pct": round(away_win * 100, 2),
            "expected_home_goals": round(lh, 3),
            "expected_away_goals": round(la, 3),
            "most_likely_score": f"{most_likely[0]}-{most_likely[1]}"
        }


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    df = load_champions_data()

    predictor = PoissonPredictor(df)

    print("\n🧪 Test:")

    result = predictor.predict_match("Real Madrid", "Barcelona")

    print(result)