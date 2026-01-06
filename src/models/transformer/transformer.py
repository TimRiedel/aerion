import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TrajectoryTransformer(nn.Module):
    """
    Autoregressive transformer model for trajectory prediction with encoder-decoder architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        max_input_len: int = 15,
        max_output_len: int = 120,
        batch_first: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # Input embedding: map features to d_model dimension
        self.src_embedding = nn.Linear(input_dim, d_model)
        self.tgt_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.src_pos_encoding = PositionalEncoding(d_model, max_len=max_input_len, dropout=dropout)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len=max_output_len, dropout=dropout)
        
        # Transformer encoder-decoder
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
        )
        
        # Output projection: map back to feature space
        self.output_projection = nn.Linear(d_model, input_dim)
        
    def forward(self, src, tgt, tgt_mask=None, tgt_pad_mask=None):
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)

        src_emb = self.src_pos_encoding(src_emb)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)
        
        # Transformer pass
        output = self.transformer(
            src=src_emb, 
            tgt=tgt_emb, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        
        return self.output_projection(output)

