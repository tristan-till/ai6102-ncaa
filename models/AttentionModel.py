import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, feature_dim=116, hidden_dim=128, num_heads=4):
        super(AttentionModel, self).__init__()
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.winner_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.odds_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        team1 = x[:, 0]
        team2 = x[:, 1]
        
        team1_encoded = self.feature_encoder(team1)
        team2_encoded = self.feature_encoder(team2)
        
        team1_encoded = team1_encoded.unsqueeze(1)
        team2_encoded = team2_encoded.unsqueeze(1)
        
        attn_t1_to_t2, _ = self.cross_attention(team1_encoded, team2_encoded, team2_encoded)
        
        attn_t2_to_t1, _ = self.cross_attention(team2_encoded, team1_encoded, team1_encoded)
        
        combined = torch.cat([
            attn_t1_to_t2.squeeze(1),
            attn_t2_to_t1.squeeze(1)
        ], dim=1) 
        
        winner_prob = self.winner_head(combined)
        odds_prob = self.odds_head(combined)
        
        return torch.cat([winner_prob, odds_prob], dim=1)