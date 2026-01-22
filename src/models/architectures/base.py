from abc import ABC, abstractmethod
import math
import torch
from torch import Tensor, nn


class TrajectoryBackbone(ABC):
    """
    Abstract base class for trajectory encoder-decoder backbones.
    """
    
    @abstractmethod
    def encode(self, input_traj: Tensor) -> Tensor:
        """
        Encode the input trajectory.
        
        Args:
            input_traj: Input trajectory tensor [batch, seq_len, encoder_input_dim]
            
        Returns:
            Memory tensor [batch, seq_len, d_model]
        """
        ...
    
    @abstractmethod
    def decode(self, dec_in_traj: Tensor, memory: Tensor, causal_mask: Tensor) -> Tensor:
        """
        Decode the output trajectory.
        
        Args:
            dec_in_traj: Decoder input tensor [batch, horizon_len, decoder_input_dim]
            memory: Encoder memory tensor [batch, seq_len, d_model]
            causal_mask: Causal mask for autoregressive decoding
            
        Returns:
            Predicted trajectory tensor [batch, horizon_len, decoder_input_dim]
        """
        ...
    
    @abstractmethod
    def forward(
        self, 
        input_traj: Tensor, 
        dec_in_traj: Tensor, 
        causal_mask: Tensor, 
        target_pad_mask: Tensor
    ) -> Tensor:
        """
        Full forward pass through encoder and decoder.
        
        Args:
            input_traj: Input trajectory tensor [batch, seq_len, encoder_input_dim]
            dec_in_traj: Decoder input tensor [batch, horizon_len, decoder_input_dim]
            causal_mask: Causal mask for autoregressive decoding
            target_pad_mask: Padding mask for variable-length targets
            
        Returns:
            Predicted trajectory tensor [batch, horizon_len, decoder_input_dim]
        """
        ...


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""
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