import torch
import torch.nn as nn
from typing import Optional

class ContextAwareTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with optional cross-attention to context embeddings.
    
    Structure:
    1. Self-attention on trajectory
    2. Cross-attention to flightinfo context (if enabled)
    3. Feed-forward network
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        batch_first: bool = True,
        use_flightinfo: bool = True,
    ):
        super().__init__()
        
        self.use_flightinfo = use_flightinfo
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.dropout_sattn = nn.Dropout(dropout)
        self.norm_sattn = nn.LayerNorm(d_model)
        
        # Cross-attention to flightinfo (optional)
        if self.use_flightinfo:
            self.flightinfo_cross_attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=batch_first
            )
            self.dropout_flightinfo = nn.Dropout(dropout)
            self.norm_flightinfo = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        
    
    def forward(
        self,
        traj_emb: torch.Tensor,
        flightinfo_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            traj: Trajectory embeddings [batch, seq_len, d_model]
            flightinfo_emb: Encoded flightinfo [batch, 1, d_model] or None
            
        Returns:
            Updated trajectory embeddings [batch, seq_len, d_model]
        """
        # 1. Self-attention on trajectory
        self_attention_out, _ = self.self_attention(traj_emb, traj_emb, traj_emb)
        traj_emb += self.dropout_sattn(self_attention_out)
        traj_emb = self.norm_sattn(traj_emb)
        
        # 2. Cross-attention to flightinfo (if enabled and provided)
        if self.use_flightinfo and flightinfo_emb is not None:
            cross_attention_out, _ = self.flightinfo_cross_attn(
                query=traj_emb,
                key=flightinfo_emb,
                value=flightinfo_emb
            )
            traj_emb += self.dropout_flightinfo(cross_attention_out)
            traj_emb = self.norm_flightinfo(traj_emb)
        
        # 3. Feed-forward network
        ffn_out = self.feedforward(traj_emb)
        traj_emb = traj_emb + self.dropout_ffn(ffn_out)
        traj_emb = self.norm_ffn(traj_emb)
        
        return traj_emb