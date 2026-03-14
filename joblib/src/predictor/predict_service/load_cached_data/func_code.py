# first line: 31
@joblib.Memory(BASE_DIR).cache()
def load_cached_data(csv_path: str):
    """Carga datos con cache persistente"""
    print(f"📊 Cargando datos desde: {csv_path}")
    df = load_champions_data(csv_path)
    
    # Normalizar nombres de columnas si es necesario
    if "HomeTeam" not in df.columns:
        df = df.rename(columns={
            "home_team": "HomeTeam",
            "away_team": "AwayTeam",
            "home_goals": "HomeGoals", 
            "away_goals": "AwayGoals"
        })
    
    teams = list(set(df["HomeTeam"]).union(set(df["AwayTeam"])))
    print(f"✅ Dataset cargado: {len(df)} partidos, {len(teams)} equipos")
    
    return df, list(teams)
