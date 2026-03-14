import pandas as pd


# DETECTAR COLUMNAS

def detect_columns(df):

    home_col = "home_team" if "home_team" in df.columns else "HomeTeam"
    away_col = "away_team" if "away_team" in df.columns else "AwayTeam"
    hg_col = "home_goals" if "home_goals" in df.columns else "HomeGoals"
    ag_col = "away_goals" if "away_goals" in df.columns else "AwayGoals"

    return home_col, away_col, hg_col, ag_col


# BUSCAR EQUIPOS EN PREGUNTA
def find_teams_in_question(df: pd.DataFrame, question: str):

    home_col, away_col, _, _ = detect_columns(df)

    q = question.lower()

    teams = set(df[home_col].unique()).union(df[away_col].unique())

    mentioned = []

    for team in teams:

        team_lower = team.lower()

        if team_lower in q:
            mentioned.append(team)

        else:
            # coincidencia parcial
            words = team_lower.split()

            for w in words:
                if len(w) > 3 and w in q:
                    mentioned.append(team)
                    break

    return mentioned


# HISTORIAL ENTRE DOS EQUIPOS
def build_head_to_head_stats(df: pd.DataFrame, teams: list):

    home_col, away_col, hg_col, ag_col = detect_columns(df)

    if len(teams) < 2:
        return ""

    team1 = teams[0]
    team2 = teams[1]

    matches = df[
        ((df[home_col] == team1) & (df[away_col] == team2)) |
        ((df[home_col] == team2) & (df[away_col] == team1))
    ]

    if matches.empty:
        return ""

    t1_wins = 0
    t2_wins = 0
    draws = 0

    for r in matches.itertuples():

        hg = int(getattr(r, hg_col))
        ag = int(getattr(r, ag_col))

        if hg > ag:
            winner = getattr(r, home_col)
        elif ag > hg:
            winner = getattr(r, away_col)
        else:
            winner = "draw"

        if winner == team1:
            t1_wins += 1
        elif winner == team2:
            t2_wins += 1
        else:
            draws += 1

    stats = f"""
Historial {team1} vs {team2}

{team1} ganó: {t1_wins}
{team2} ganó: {t2_wins}
Empates: {draws}
Partidos totales: {len(matches)}
"""

    return stats


# PARTIDOS RELEVANTES
def find_relevant_matches(df: pd.DataFrame, teams: list, max_rows=10):

    home_col, away_col, _, _ = detect_columns(df)

    if not teams:
        return df.head(max_rows)

    matches = df[
        (df[home_col].isin(teams)) |
        (df[away_col].isin(teams))
    ]

    if matches.empty:
        return df.head(max_rows)

    return matches.head(max_rows)


# CONTEXTO PARA EL LLM
def build_match_context(df: pd.DataFrame, question: str, max_rows: int = 10):

    home_col, away_col, hg_col, ag_col = detect_columns(df)

    teams = find_teams_in_question(df, question)

    stats = build_head_to_head_stats(df, teams)

    matches = find_relevant_matches(df, teams, max_rows)

    lines = []

    for r in matches.itertuples():

        line = f"{getattr(r, home_col)} {getattr(r, hg_col)}-{getattr(r, ag_col)} {getattr(r, away_col)}"

        if "season" in df.columns:
            season = getattr(r, "season", "")
            if season:
                line += f" ({season})"

        lines.append(line)

    matches_text = "\n".join(lines)

    context = f"""
Datos del dataset Champions League

{stats}

Partidos relevantes:

{matches_text}
"""

    return context