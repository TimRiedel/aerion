from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor


class TrajectoryBackbone(ABC):
    """
    Abstract base class for trajectory encoder-decoder backbones.

    Tensor shapes are dimension-agnostic:
    - Single-agent: [B, T, F]
    - Multi-agent:  [B, T, N, F]
    """
    
    @abstractmethod
    def encode(self, input_traj: Tensor, target_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Encode the input trajectory.
        
        Args:
            input_traj: Input trajectory tensor [B, T_in, F] or [B, T_in, N, F].
            target_padding_mask: Optional padding mask:
                - Single-agent: typically ``None`` (no agent dimension).
                - Multi-agent:  [B, H, N] — True = padding, used (e.g. by AgentFormer)
                  to derive an encoder source key padding mask for padded agent slots.
            
        Returns:
            Memory tensor with the same leading dims as ``input_traj``.
        """
        ...
    
    @abstractmethod
    def decode(
        self,
        dec_in_traj: Tensor,
        memory: Tensor,
        causal_mask: Optional[Tensor] = None,
        target_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode the output trajectory.
        
        Args:
            dec_in_traj: Decoder input tensor [B, H, F] or [B, H, N, F].
            memory: Encoder memory tensor with the same leading dims as returned
                by ``encode`` (e.g. [B, T_in, d_model] or [B, T_in, N, d_model]).
            causal_mask: Causal (attention) mask used for autoregressive decoding:
                - Single-agent: [H, H] square subsequent mask.
                - Multi-agent:  [H*N, H*N] block-causal mask (time-major, agents interleaved).
            target_padding_mask: Optional padding mask:
                - Single-agent: [B, H] — True = padding (PyTorch ``tgt_key_padding_mask``).
                - Multi-agent:  [B, H, N] — True = padding (flattened internally to [B, H*N]).
            
        Returns:
            Predicted trajectory tensor with the same leading dims as ``dec_in_traj``.
        """
        ...
    
    @abstractmethod
    def forward(
        self, 
        input_traj: Tensor, 
        dec_in_traj: Tensor, 
        causal_mask: Optional[Tensor] = None, 
        target_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Full forward pass through encoder and decoder.
        
        Args:
            input_traj: Input trajectory tensor [B, T_in, F] or [B, T_in, N, F].
            dec_in_traj: Decoder input tensor [B, H, F] or [B, H, N, F].
            causal_mask: Causal (attention) mask used for autoregressive decoding:
                - Single-agent: [H, H] square subsequent mask.
                - Multi-agent:  [H*N, H*N] block-causal mask (time-major, agents interleaved).
            target_padding_mask: Optional padding mask:
                - Single-agent: [B, H] — True = padding (PyTorch ``tgt_key_padding_mask``).
                - Multi-agent:  [B, H, N] — True = padding (flattened internally to [B, H*N]).
            
        Returns:
            Predicted trajectory tensor with the same leading dims as ``dec_in_traj``.
        """
        ...