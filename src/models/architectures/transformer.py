from typing import Optional

import torch
import torch.nn as nn

from .common import PositionalEncoding, TrajectoryBackbone
from .layers import CustomTransformerDecoderLayer, CustomTransformerEncoderLayer


class PytorchTrajectoryTransformer(nn.Module, TrajectoryBackbone):
    """Single Agent Trajectory Transformer with autoregressive encoder-decoder architecture."""
    
    def __init__(
        self,
        encoder_input_dim: int = 8,
        decoder_input_dim: int = 5,
        output_dim: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_input_len: int = 10,
        max_output_len: int = 80,
        batch_first: bool = True,
    ):
        super().__init__()
        
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # Input embedding
        self.input_embedding = nn.Linear(encoder_input_dim, d_model)
        self.input_pos_encoding = PositionalEncoding(d_model, max_len=max_input_len, dropout=dropout)
        
        # Decoder input embedding
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
            activation='gelu',
        )
        self.output_projection = nn.Linear(d_model, output_dim)

    def encode(self, input_traj: torch.Tensor) -> torch.Tensor:
        input_emb = self.input_embedding(input_traj)
        input_emb = self.input_pos_encoding(input_emb)
        return self.transformer.encoder(input_emb)

    def decode(
        self, 
        dec_in_traj: torch.Tensor, 
        memory: torch.Tensor, 
        causal_mask: Optional[torch.Tensor] = None,
        target_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dec_in_emb = self.dec_in_embedding(dec_in_traj)
        dec_in_emb = self.dec_in_pos_encoding(dec_in_emb)
        
        output = self.transformer.decoder(
            tgt=dec_in_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_pad_mask
        )
        return self.output_projection(output)
        
    def forward(
        self, 
        input_traj: torch.Tensor, 
        dec_in_traj: torch.Tensor, 
        causal_mask: Optional[torch.Tensor] = None, 
        target_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        memory = self.encode(input_traj)
        output = self.decode(dec_in_traj, memory, causal_mask=causal_mask, target_pad_mask=target_pad_mask)
        return output


class TrajectoryTransformer(nn.Module, TrajectoryBackbone):
    """Single Agent Trajectory Transformer with autoregressive encoder-decoder architecture."""
    
    def __init__(
        self,
        encoder_input_dim: int = 8,
        decoder_input_dim: int = 5,
        output_dim: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_input_len: int = 10,
        max_output_len: int = 80,
        batch_first: bool = True,
    ):
        super().__init__()
        
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # Input embedding
        self.input_embedding = nn.Linear(encoder_input_dim, d_model)
        self.input_pos_encoding = PositionalEncoding(d_model, max_len=max_input_len, dropout=dropout)

        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=batch_first,
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder input embedding
        self.dec_in_embedding = nn.Linear(decoder_input_dim, d_model)
        self.dec_in_pos_encoding = PositionalEncoding(d_model, max_len=max_output_len, dropout=dropout)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            CustomTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=batch_first,
            )
            for _ in range(num_decoder_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, output_dim)
    
    def encode(
        self,
        input_traj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_traj: Input trajectory [batch, seq_len, encoder_input_dim]
            
        Returns:
            Encoded trajectory tensor [batch, seq_len, d_model]
        """
        traj_emb = self.input_embedding(input_traj)
        traj_emb = self.input_pos_encoding(traj_emb)
        
        for layer in self.encoder_layers:
            traj_emb = layer(traj_emb)
        
        return traj_emb
    
    def decode(
        self,
        dec_in_traj: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        target_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            dec_in_traj: Decoder input [batch, horizon_len, decoder_input_dim]
            memory: Encoder memory [batch, seq_len, d_model]
            causal_mask: Causal mask for autoregressive decoding
            target_pad_mask: Padding mask for variable-length targets [batch, horizon_len] (True = padded)
            
        Returns:
            Predicted deltas [batch, horizon_len, output_dim]
        """
        dec_in_emb = self.dec_in_embedding(dec_in_traj)
        dec_in_emb = self.dec_in_pos_encoding(dec_in_emb)
        
        for layer in self.decoder_layers:
            dec_in_emb = layer(
                target_emb=dec_in_emb,
                memory=memory,
                target_mask=causal_mask,
                target_key_padding_mask=target_pad_mask,
            )
        
        return self.output_projection(dec_in_emb)
    
    def forward(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        target_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_traj: Input trajectory [batch, seq_len, encoder_input_dim]
            dec_in_traj: Decoder input [batch, horizon_len, decoder_input_dim]
            causal_mask: Causal mask for autoregressive decoding [batch, horizon_len, horizon_len]
            target_pad_mask: Padding mask for variable-length targets [batch, horizon_len]
            
        Returns:
            Predicted deltas [batch, horizon_len, output_dim]
        """
        memory = self.encode(input_traj)
        output = self.decode(
            dec_in_traj, 
            memory, 
            causal_mask=causal_mask, 
            target_pad_mask=target_pad_mask,
        )
        return output
