import torch
import torch.nn as nn
import math
from typing import Optional, Dict
from omegaconf import DictConfig

from .base import PositionalEncoding, TrajectoryBackbone
from .layers import ContextAwareTransformerEncoderLayer, ContextAwareTransformerDecoderLayer
from .encoders import FlightInfoEncoder


class Aerion(nn.Module, TrajectoryBackbone):
    """Context-aware transformer for trajectory prediction. """
    
    def __init__(
        self,
        encoder_input_dim: int = 6,
        decoder_input_dim: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_input_len: int = 15,
        max_output_len: int = 120,
        batch_first: bool = True,
        contexts_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.contexts_cfg = contexts_cfg or {}
        
        # Determine which contexts are enabled
        self.use_flightinfo = self._is_context_enabled("flightinfo")
        
        # Trajectory input embedding
        self.input_embedding = nn.Linear(encoder_input_dim, d_model)
        self.input_pos_encoding = PositionalEncoding(d_model, max_len=max_input_len, dropout=dropout)
        self._initialize_context_encoders()
        
        # Context-aware encoder layers
        self.encoder_layers = nn.ModuleList([
            ContextAwareTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=batch_first,
                use_flightinfo=self.use_flightinfo,
            )
            for _ in range(num_encoder_layers)
        ])
        
        self.dec_in_embedding = nn.Linear(decoder_input_dim, d_model)
        self.dec_in_pos_encoding = PositionalEncoding(d_model, max_len=max_output_len, dropout=dropout)
        
        # Context-aware decoder layers
        self.decoder_layers = nn.ModuleList([
            ContextAwareTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=batch_first,
                use_flightinfo=self.use_flightinfo,
            )
            for _ in range(num_decoder_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, decoder_input_dim)
    
    def _is_context_enabled(self, name: str) -> bool:
        """Check if a context is enabled in the configuration."""
        return self.contexts_cfg.get(name, {}).get("enabled", False)

    def _initialize_context_encoders(self):
        if self.use_flightinfo:
            flightinfo_input_dim = len(self.contexts_cfg.get("flightinfo", {}).get("features", []))
            if flightinfo_input_dim == 0:
                raise ValueError("Flightinfo input dimension is 0, but at least 1 feature is required")
            self.flightinfo_encoder = FlightInfoEncoder(d_model=self.d_model, input_dim=flightinfo_input_dim)
    
    def encode(
        self,
        input_traj: torch.Tensor,
        contexts: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_traj: Input trajectory [batch, seq_len, encoder_input_dim]
            contexts: Dictionary of context tensors {"flightinfo": [batch, num_features], ...}
            
        Returns:
            Context-enriched memory tensor [batch, seq_len, d_model]
        """
        contexts = contexts or {}
        
        # Embed trajectory
        traj_emb = self.input_embedding(input_traj)
        traj_emb = self.input_pos_encoding(traj_emb)
        
        # Encode contexts
        flightinfo_emb = None
        if self.use_flightinfo and "flightinfo" in contexts:
            flightinfo_emb = self.flightinfo_encoder(contexts["flightinfo"])
        
        # Process through context-aware encoder layers
        for layer in self.encoder_layers:
            traj_emb = layer(traj_emb, flightinfo_emb=flightinfo_emb)
        
        return traj_emb
    
    def decode(
        self,
        dec_in_traj: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        target_pad_mask: Optional[torch.Tensor] = None,
        flightinfo_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            dec_in_traj: Decoder input [batch, horizon_len, decoder_input_dim]
            memory: Encoder memory [batch, seq_len, d_model]
            causal_mask: Causal mask for autoregressive decoding
            target_pad_mask: Padding mask for variable-length targets [batch, horizon_len] (True = padded)
            flightinfo_emb: Encoded flightinfo [batch, 1, d_model] or None
            
        Returns:
            Predicted trajectory [batch, horizon_len, decoder_input_dim]
        """
        dec_in_emb = self.dec_in_embedding(dec_in_traj)
        dec_in_emb = self.dec_in_pos_encoding(dec_in_emb)
        
        for layer in self.decoder_layers:
            dec_in_emb = layer(
                target_emb=dec_in_emb,
                memory=memory,
                target_mask=causal_mask,
                target_key_padding_mask=target_pad_mask,
                flightinfo_emb=flightinfo_emb,
            )
        
        return self.output_projection(dec_in_emb)
    
    def forward(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        target_pad_mask: Optional[torch.Tensor] = None,
        contexts: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_traj: Input trajectory [batch, seq_len, encoder_input_dim]
            dec_in_traj: Decoder input [batch, horizon_len, decoder_input_dim]
            causal_mask: Causal mask for autoregressive decoding [batch, horizon_len, horizon_len]
            target_pad_mask: Padding mask for variable-length targets [batch, horizon_len]
            contexts: Dictionary of context tensors
            
        Returns:
            Predicted trajectory [batch, horizon_len, decoder_input_dim]
        """
        contexts = contexts or {}
        
        # Encode input trajectory with contexts
        memory = self.encode(input_traj, contexts=contexts)
        
        # Encode contexts for decoder
        flightinfo_emb = None
        if self.use_flightinfo and "flightinfo" in contexts:
            flightinfo_emb = self.flightinfo_encoder(contexts["flightinfo"])
        
        # Decode with context-aware decoder
        output = self.decode(
            dec_in_traj, 
            memory, 
            causal_mask=causal_mask, 
            target_pad_mask=target_pad_mask,
            flightinfo_emb=flightinfo_emb,
        )
        return output
