import pandas as pd

df = pd.read_csv("data/champions_league_clean.csv")

# limpiar nombres de columnas
df.columns = df.columns.str.strip()

print("Filas originales:", len(df))

# seleccionar columnas importantes
df = df[[
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "season",
    "Round"
]]

# convertir goles a número
df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")

# eliminar filas sin goles
df = df.dropna(subset=["home_goals", "away_goals"])

print("Filas después de limpiar:", len(df))

print(df.head())

# guardar dataset limpio
df.to_csv("data/matches_clean.csv", index=False)

print("Dataset limpio guardado en data/matches_clean.csv")