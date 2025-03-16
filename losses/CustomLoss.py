import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.criterion = nn.BCELoss()
        self.winner_weight=1.0
        self.odds_weight=1.0
        
    def forward(self, pred, target):
        # winner_loss = F.binary_cross_entropy(
        #     pred[:, 0].view(-1),
        #     target[:, 0].view(-1)
        # )
        
        # odds_loss = F.mse_loss(
        #     pred[:, 1].view(-1), 
        #     target[:, 1].view(-1)
        # )
        combined_loss = F.binary_cross_entropy(pred, target)
        # combined_loss = self.winner_weight * winner_loss + self.odds_weight * odds_loss
        return combined_loss