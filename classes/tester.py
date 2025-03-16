import os

from tqdm import tqdm
import numpy as np
import torch

from utils import const

def test(model, test_loader):
    train_winner_acc = 0.0
    pred_winner_acc = 0.0
    
    for _, (data, target) in enumerate(test_loader):
        # Randomly flip teams to avoid order bias (with 50% probability)
        if np.random.random() < 0.5:
            data = torch.stack([data[:, 1], data[:, 0]], dim=1)
            target[:, 0] = 1.0 - target[:, 0]
            target[:, 1] = 1.0 - target[:, 1]
        
        data, target = data.to(const.DEVICE), target.to(const.DEVICE)
        
        output = model(data)
        
        pred_winner = (output[:, 0] > 0.5).float()
        true_winner = (target[:, 0] > 0.5).float()
        accuracy = (pred_winner == true_winner).float().mean().item()
        train_winner_acc += accuracy
        
        pred_odds = (output[:, 1] > 0.5).float()
        true_odds = (target[:, 1] > 0.5).float()
        pred_accuracy = (pred_odds == true_odds).float().mean().item()
        pred_winner_acc += pred_accuracy
        
        # odds_mae = torch.abs(output[:, 1] - target[:, 1]).mean().item()
        # train_odds_mae += odds_mae
                
    train_winner_acc /= len(test_loader)
    pred_winner_acc /= len(test_loader)
    print(f"Winner Acc: {train_winner_acc:.4f}, Odds Acc: {pred_winner_acc:.4f}")