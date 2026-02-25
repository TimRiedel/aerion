from abc import ABC, abstractmethod

from torch import Tensor


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
    def decode(self, dec_in_traj: Tensor, memory: Tensor, causal_mask: Tensor, target_pad_mask: Tensor = None) -> Tensor:
        """
        Decode the output trajectory.
        
        Args:
            dec_in_traj: Decoder input tensor [batch, horizon_len, decoder_input_dim]
            memory: Encoder memory tensor [batch, seq_len, d_model]
            causal_mask: Causal mask for autoregressive decoding
            target_pad_mask: Padding mask for variable-length targets
            
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
        target_pad_mask: Tensor = None
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