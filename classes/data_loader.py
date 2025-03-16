import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import preprocessing.clean as clean
from utils import const

def load():
    if not os.path.exists("cache/results.csv") or not os.path.exists("cache/teams.csv"):
        clean.run()
    odds = pd.read_csv("cache/results.csv")
    odds = odds.drop(["win_team2", "win_pct_team2"], axis=1)
    teams = pd.read_csv("cache/teams.csv")
    teams = teams.set_index("SchoolY").T.to_dict()
    odds = np.array(odds)
    X_temp = odds[:, :2]
    X = []
    y = []

    for i, (t1, t2) in enumerate(X_temp):
        if t1 not in teams or t2 not in teams:
            # print(f"Skipping row {i}: {t1}, {t2} - one or both teams not found")
            continue

        team1_values = np.array(list(teams[t1].values()))
        team2_values = np.array(list(teams[t2].values()))
        new_row_x = [team1_values, team2_values]
        X.append(new_row_x)
        y.append(odds[i, 2:])

    X = np.array(X)
    y = np.array(y, dtype=np.float64)
    
    X = torch.tensor(X, dtype=torch.float32).to(const.DEVICE)
    y = torch.tensor(y, dtype=torch.float32).to(const.DEVICE)
    y = (y > 0.5).float()
    
    dataset = TensorDataset(X, y)
    dataset_size = len(dataset)
    train_size = int(const.TRAIN_SPLIT * dataset_size)
    val_size = int(const.VAL_SPLIT * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=const.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=const.BATCH_SIZE, shuffle=False, num_workers=const.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=const.BATCH_SIZE, shuffle=False, num_workers=const.NUM_WORKERS)
    return train_loader, val_loader, test_loader