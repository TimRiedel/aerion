from typing import Optional

import torch
import torch.nn as nn


class CustomTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer.
    
    Structure:
    1. Self-attention on trajectory
    2. Feed-forward network
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.dropout_sattn = nn.Dropout(dropout)
        self.norm_sattn = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        
    
    def forward(
        self,
        traj_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            traj: Trajectory embeddings [batch, seq_len, d_model]
            
        Returns:
            Updated trajectory embeddings [batch, seq_len, d_model]
        """
        # 1. Self-attention on trajectory
        self_attention_out, _ = self.self_attention(traj_emb, traj_emb, traj_emb)
        traj_emb += self.dropout_sattn(self_attention_out)
        traj_emb = self.norm_sattn(traj_emb)
        
        
        # 2. Feed-forward network
        ffn_out = self.feedforward(traj_emb)
        traj_emb = traj_emb + self.dropout_ffn(ffn_out)
        traj_emb = self.norm_ffn(traj_emb)
        
        return traj_emb


class CustomTransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    
    Structure:
    1. Causal self-attention on target sequence
    2. Cross-attention to encoder memory
    3. Feed-forward network
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        
        # Self-attention (causal for autoregressive decoding)
        self.self_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.dropout_sattn = nn.Dropout(dropout)
        self.norm_sattn = nn.LayerNorm(d_model)
        
        # Cross-attention to encoder memory
        self.memory_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.dropout_memory = nn.Dropout(dropout)
        self.norm_memory = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
    
    def forward(
        self,
        target_emb: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            target_emb: Decoder input (target) embeddings [batch, seq_len, d_model]
            memory: Encoder memory [batch, src_len, d_model]
            target_mask: Causal mask for self-attention [seq_len, seq_len]
            target_key_padding_mask: Padding mask for target [batch, seq_len] (True = padded)
            
        Returns:
            Updated target embeddings [batch, seq_len, d_model]
        """
        # 1. Causal self-attention on target sequence
        self_attention_out, _ = self.self_attention(
            query=target_emb,
            key=target_emb,
            value=target_emb,
            attn_mask=target_mask,
            key_padding_mask=target_key_padding_mask,
        )
        target_emb += self.dropout_sattn(self_attention_out)
        target_emb = self.norm_sattn(target_emb)
        
        # 2. Cross-attention to encoder memory
        memory_attention_out, _ = self.memory_cross_attn(
            query=target_emb,
            key=memory,
            value=memory,
        )
        target_emb += self.dropout_memory(memory_attention_out)
        target_emb = self.norm_memory(target_emb)
        
        # 3. Feed-forward network
        ffn_out = self.feedforward(target_emb)
        target_emb += self.dropout_ffn(ffn_out)
        target_emb = self.norm_ffn(target_emb)
        
        return target_emb