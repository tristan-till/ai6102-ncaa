import numpy as np
import pandas as pd
import torch

from models.AttentionModel import AttentionModel
from preprocessing.rename import get_map
import utils.const as const

def get_matchups():
    id_to_team, _ = get_map()
    team_ids = id_to_team.keys()
    team_ids = [team_id for team_id in team_ids if team_id.endswith("_2025")]
    male_team_ids = [team_id for team_id in team_ids if team_id.startswith("1")]
    female_team_ids = [team_id for team_id in team_ids if team_id.startswith("3")]
    matchups = []
    for team_id in male_team_ids:
        for other_team_id in male_team_ids:
            if team_id != other_team_id:
                matchups.append([team_id, other_team_id])
    for team_id in female_team_ids:
        for other_team_id in female_team_ids:
            if team_id != other_team_id:
                matchups.append([team_id, other_team_id])
    return matchups

def predict_kaggle():
    
    matchups = get_matchups()
    teams = pd.read_csv("cache/teams.csv")
    teams = teams.set_index("SchoolY").T.to_dict()
    X = []
    model = AttentionModel()
    model.load_state_dict(torch.load(f"checkpoints/{const.RUN_NAME}_best.pt", weights_only=True))
    model.eval()
    
    winner_preds = pd.DataFrame(columns=["ID", "Pred"])
    odds_preds = pd.DataFrame(columns=["ID", "Pred"])
    for i, (t1, t2) in enumerate(matchups):
        if t1 not in teams or t2 not in teams:
            print(f"Skipping row {i}: {t1}, {t2} - one or both teams not found")
            continue

        team1_values = torch.Tensor(list(teams[t1].values()))
        team2_values = torch.Tensor(list(teams[t2].values()))
        inp = torch.stack([team1_values, team2_values], dim=0).unsqueeze(0)
        with torch.no_grad():
            out = model(inp)
        id = f"2025_{t1.split('_')[0]}_{t2.split('_')[0]}"
        winner_preds.loc[len(winner_preds)] = [id, out[0][0].item()]
        odds_preds.loc[len(odds_preds)] = [id, out[0][1].item()]
        
        if i > 25:
            break

    winner_preds.to_csv("predictions/winner_preds.csv", index=False)
    odds_preds.to_csv("predictions/odds_preds.csv", index=False)