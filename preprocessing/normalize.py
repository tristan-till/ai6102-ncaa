import numpy as np

def normalize_polls(df):
    
    teams = df["SchoolY"]
    df = df.drop(["SchoolY"], axis=1)
    df = df.astype(float)
    max_val = max(df.max().values)
    df = (max_val + 1 - df) / max_val
    df = df.fillna(0)
    df["SchoolY"] = teams
    return df

def normalize_ratings(df):
    for col in [col for col in df.columns if col != "SchoolY"]:
        df[col] = df[col].astype(float)
        col_min = df[col].min()
        col_max = df[col].max()
        col_norm = (df[col] - col_min) / (col_max - col_min)
        df[col] = col_norm
    # df["ratings_L"] = 1 - df["ratings_L"]
    df = df.fillna(0.5)
    return df

def normalize_teams_adv(df):
    for col in [col for col in df.columns if col != "SchoolY"]:
        df[col] = df[col].astype(float)
        col_min = df[col].min()
        col_max = df[col].max()
        col_norm = (df[col] - col_min) / (col_max - col_min)
        df[col] = col_norm
    df = df.fillna(0.5)
    return df

def normalize_teams_base(df):
    teams = df["SchoolY"]
    df = df.drop(["SchoolY"], axis=1)
    df = df.astype(float)
    df["SchoolY"] = teams
    mp = df["base_MP"]
    
    games_avg = df["base_G"].mean()
    mp_avg = mp.mean()
    avg_min_per_game = mp_avg / games_avg
    
    df["base_MP"] = np.where(df["base_MP"].isna(), df["base_G"] * avg_min_per_game, df["base_MP"])
    for col in ["FG","FGA","3P","3PA","FT","FTA","ORB","TRB","AST",
                "STL","BLK","TOV","PF"]:
        df[f"base_{col}"] = df[f"base_{col}"] / df["base_MP"]
    for col in [col for col in df.columns if col != "SchoolY"]:
        col_min = df[col].min()
        col_max = df[col].max()
        col_norm = (df[col] - col_min) / (col_max - col_min)
        df[col] = col_norm
    df = df.fillna(0.5)
    return df

def normalize_opps_adv(df):
    for col in [col for col in df.columns if col != "SchoolY"]:
        df[col] = df[col].astype(float)
        col_min = df[col].min()
        col_max = df[col].max()
        col_norm = (df[col] - col_min) / (col_max - col_min)
        df[col] = col_norm
    df = df.fillna(0.5)
    return df

def normalize_opps_base(df):
    teams = df["SchoolY"]
    df = df.drop(["SchoolY"], axis=1)
    df = df.astype(float)
    df["SchoolY"] = teams
    mp = df["opp_base_MP"]
    games_avg = df["opp_base_G"].mean()
    mp_avg = mp.mean()
    avg_min_per_game = mp_avg / games_avg
    
    df["opp_base_MP"] = np.where(df["opp_base_MP"].isna(), df["opp_base_G"] * avg_min_per_game, df["opp_base_MP"])
    for col in ["opp_base_FG","opp_base_FGA","opp_base_3P","opp_base_3PA","opp_base_FT","opp_base_FTA","opp_base_ORB",
                "opp_base_TRB","opp_base_AST", "opp_base_STL","opp_base_BLK","opp_base_TOV","opp_base_PF"]:
        df[f"{col}"] = df[f"{col}"] / df["opp_base_MP"]
    for col in [col for col in df.columns if col != "SchoolY"]:
        col_min = df[col].min()
        col_max = df[col].max()
        col_norm = (df[col] - col_min) / (col_max - col_min)
        df[col] = col_norm
    df = df.fillna(0.5)
    return df