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
        encoder_input_dim: int = 6,
        decoder_input_dim: int = 3,
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
        
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        self.input_embedding = nn.Linear(encoder_input_dim, d_model)
        self.input_pos_encoding = PositionalEncoding(d_model, max_len=max_input_len, dropout=dropout)
        
        self.dec_in_embedding = nn.Linear(decoder_input_dim, d_model)
        self.dec_in_pos_encoding = PositionalEncoding(d_model, max_len=max_output_len, dropout=dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
        )
        
        self.output_projection = nn.Linear(d_model, decoder_input_dim)

    def encode(self, input_traj: torch.Tensor):
        input_emb = self.input_embedding(input_traj)
        input_emb = self.input_pos_encoding(input_emb)
        return self.transformer.encoder(input_emb)

    def decode(self, dec_in_traj: torch.Tensor, memory: torch.Tensor, causal_mask: Optional[torch.Tensor] = None):
        dec_in_emb = self.dec_in_embedding(dec_in_traj)
        dec_in_emb = self.dec_in_pos_encoding(dec_in_emb)
        
        output = self.transformer.decoder(
            tgt=dec_in_emb, 
            memory=memory, 
            tgt_mask=causal_mask
        )
        return self.output_projection(output)
        
    def forward(self, input_traj: torch.Tensor, dec_in_traj: torch.Tensor, causal_mask: Optional[torch.Tensor] = None, target_pad_mask: Optional[torch.Tensor] = None):
        memory = self.encode(input_traj)
        output = self.decode(dec_in_traj, memory, causal_mask=causal_mask)
        return output

