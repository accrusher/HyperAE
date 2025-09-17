import torch
import torch.nn as nn
import torch.nn.functional as F


class LightResidualPredictor(nn.Module):
    def __init__(self, feat_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.dropout(self.act(self.proj(x)))

class DecomposedPredictor(nn.Module):
    def __init__(self, feat_dim, dropout):
        super().__init__()
        self.predictor_pos = LightResidualPredictor(feat_dim, dropout)
        self.predictor_neg = LightResidualPredictor(feat_dim, dropout)
        
    def forward(self, Z_H):
        Z_H_pos = F.relu(Z_H)
        Z_H_neg = F.relu(-Z_H)
        
        Z_A_hat_pos = self.predictor_pos(Z_H_pos)
        Z_A_hat_neg = self.predictor_neg(Z_H_neg)
        return Z_A_hat_pos, Z_A_hat_neg
    
class OrthogonalDecomposer(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.proj_matrix = nn.Parameter(torch.eye(feat_dim))
        
    def forward(self, Z_H):
        Q, R = torch.linalg.qr(self.proj_matrix)
        proj_ortho = Q
        
        Z_H_pos = Z_H @ proj_ortho
        Z_H_neg = Z_H - Z_H_pos
        
        ortho_loss = torch.norm(Z_H_pos @ Z_H_neg.t(), p='fro') / Z_H.size(0)
        
        return Z_H_pos, Z_H_neg, ortho_loss
    