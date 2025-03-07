import os
import glob
import pandas as pd
import numpy as np
import re
from utils.aliases import ALIASES
import utils.helpers as helpers
import preprocessing.normalize as normalize
import utils.const as const
import preprocessing.rename as rename

def run(cache_results=True):
    print("loading data...")
    results = clean_res()
    teams = clean_teams()
    os.makedirs("cache", exist_ok=True)
    if cache_results:
        print("caching data...")
        results.to_csv("cache/results.csv", index=False)
        teams.to_csv("cache/teams.csv", index=False)
    return teams, results
    

def clean_res():
    df_odds = clean_op(const.OP_DATA_PATH, "temp.csv")
    df_espn = clean_espn(const.ESPN_DATA_PATH, "temp.csv")
    df = pd.concat([df_odds, df_espn], join='outer', ignore_index=True)
    df = df.dropna()
    df = rename.rename_odds(df)
    return df

def clean_teams():
    df = clean_sr(const.SR_DATA_PATH)
    df = rename.rename_teams(df)
    return df

def clean_op(folder_path, save_path):
    pattern = os.path.join(folder_path, "**", "*")
    df_combined = pd.DataFrame()
    for file_path in glob.glob(pattern, recursive=True):
        if os.path.isfile(file_path): 
            df = pd.read_csv(file_path)
            odds_df = pd.DataFrame(columns = ["team1", "team2", "win_team1", "win_team2", "win_pct_team1", "win_pct_team2"])
            df = df[df["team2"] != "TX Pan American"]
            df = df[df["score"].notna()]
            df = df[df["odds_team1"] != "N/A"]
            df = df[df["odds_team2"] != "N/A"]
            odds_df["team1"] = [f'{"_".join(team.split(" "))}_M_{re.search(r'\b\d{4}\b', date).group()}' for date, team in zip(df["date"], df["team1"])]
            odds_df["team2"] = [f'{"_".join(team.split(" "))}_M_{re.search(r'\b\d{4}\b', date).group()}' for date, team in zip(df["date"], df["team2"])]
            odds_df["win_pct_team1"] = [1/odds for odds in df["odds_team1"]]
            odds_df["win_pct_team2"] = [1/odds for odds in df["odds_team2"]]
            odds_df["win_team1"] = [float(int(score.split("–")[0]) > int(score.split("–")[1])) for score in df["score"]]
            odds_df["win_team2"] = [float(1 - win_team1) for win_team1 in odds_df["win_team1"]]
            odds_df = odds_df.dropna()
            df_combined = pd.concat([df_combined, odds_df], join='outer', ignore_index=True)
    return df_combined

def clean_sr(folder_path):
    pattern = os.path.join(folder_path, "**", "*")
    df_combined = pd.DataFrame(columns=["SchoolY"])
    for file_path in glob.glob(pattern, recursive=True):
        if os.path.isfile(file_path):
            # print(f"Cleaning {file_path}")
            file = file_path.split("\\")[-1].split(".")[0]
            gender_idx = -3 if "ratings" in file_path else -4
            gender = file_path.split("\\")[gender_idx].capitalize()
            prefix = file_path.split("\\")[-2]
            skiprows = 2 if "polls" in file_path else 1
            df = pd.read_csv(file_path, skiprows=skiprows, header=0)
            df = df[df["School"] != "School"]
            df = df[df['School'].notna()]
            schoolY = [f"{'_'.join(team.replace('NCAA', '').strip().split(' '))}" for team in df["School"]]
            schoolY = [ALIASES[team] if team in ALIASES else team for team in schoolY]
            schoolY = [f"{team}_{gender}_{file}".lower() for team in schoolY]
            df["SchoolY"] = schoolY
            for col in ["Conf", "Rk", "School"]:
                df = df.drop([col], axis=1) if col in df.columns else df
            for i in range(21):
                col = f"Unnamed: {i}_level_1"
                df = df.drop([col], axis=1) if col in df.columns else df
            if "polls" in file_path:
                df = df.rename({col: f"{i}" for i, col in enumerate(df.columns) if "/" in col}, axis=1)
            df = df.rename({col: f"{prefix}_{col}" for col in df.columns if col != "SchoolY"}, axis=1)
            if "polls" in file_path:
                df = normalize.normalize_polls(df)
            if "ratings" in file_path:
                df = normalize.normalize_ratings(df)
            if "teams" in file_path and "advanced" in file_path:
                df = normalize.normalize_teams_adv(df)
            if "teams" in file_path and "base" in file_path:
                df = normalize.normalize_teams_base(df)
            # full_save_path = str(file_path.replace(folder_path, save_path).replace("\\", "/")).split(file)[0]
            # if not os.path.exists(full_save_path):
            #     os.makedirs(full_save_path)
            df = df.dropna()
            # df.to_csv(f"{full_save_path}/{file}.csv", index=False)
            df_combined = pd.concat([df_combined, df], join='outer', ignore_index=True)
            df_combined = df_combined.groupby("SchoolY", dropna=False).first().reset_index()
    df_combined = df_combined.fillna(0)
    # df_combined.to_csv(f"temp2.csv", index=False)
    return df_combined
            
def clean_espn(folder_path, save_path):
    pattern = os.path.join(folder_path, "**", "*")
    df_combined = pd.DataFrame()
    for file_path in glob.glob(pattern, recursive=True):
        if os.path.isfile(file_path):
            # print(f"Cleaning {file_path}")
            file = file_path.split("\\")[-1].split(".")[0]
            year = file_path.split("\\")[-2]
            df = pd.read_csv(file_path, header=0)
            df = df[df["Away Team"] != "Florida Memorial"]
            df = df[df["Away Team"] != "Fort Valley State"]
            df = df[df["Away Team"] != "Knox"]
            df = df[df["Away Team"] != "Allen University"]
            res_df = pd.DataFrame(columns = ["team1", "team2", "win_team1", "win_team2", "win_pct_team1", "win_pct_team2"])
            res_df["team1"] = [f'{"_".join(team.split(" "))}_W_{year}' for team in  df["Home Team"]]
            res_df["team2"] = [f'{"_".join(team.split(" "))}_W_{year}' for team in  df["Away Team"]]
            res_df["win_team1"] = (df["Home Score"] > df["Away Score"]).astype(np.float64)
            res_df["win_team2"] = (df["Home Score"] < df["Away Score"]).astype(np.float64)
            scores_diff = df["Home Score"] - df["Away Score"]
            win_pct = helpers.logistic_probability(scores_diff)
            res_df["win_pct_team1"] = win_pct
            res_df["win_pct_team2"] = 1 - win_pct
            res_df = res_df.dropna(axis=1, how='all')
            df_combined = pd.concat([df_combined, res_df], join='outer', ignore_index=True)
    return df_combined