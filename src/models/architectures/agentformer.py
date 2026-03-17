from typing import Optional

import torch
import torch.nn as nn

from .common import PositionalAgentEncoding, TrajectoryBackbone
from .layers.agentformer_lib import (
    AgentFormerDecoder,
    AgentFormerDecoderLayer,
    AgentFormerEncoder,
    AgentFormerEncoderLayer,
)


class TrajectoryAgentFormer(nn.Module, TrajectoryBackbone):
    """Multi-agent trajectory encoder-decoder using AgentFormer architecture with agent-aware attention."""

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
        max_num_agents: int = 15,
        gaussian_kernel: bool = False,
        sep_attn: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()

        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.num_agent = max_num_agents

        cfg = {"gaussian_kernel": gaussian_kernel, "sep_attn": sep_attn}

        self.input_embedding = nn.Linear(encoder_input_dim, d_model)
        self.input_pos_encoding = PositionalAgentEncoding(
            d_model, dropout=dropout, max_t_len=max_input_len, max_a_len=self.num_agent
        )

        self.dec_in_embedding = nn.Linear(decoder_input_dim, d_model)
        self.dec_in_pos_encoding = PositionalAgentEncoding(
            d_model, dropout=dropout, max_t_len=max_output_len, max_a_len=self.num_agent
        )

        encoder_layer = AgentFormerEncoderLayer(
            cfg, d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        self.encoder = AgentFormerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = AgentFormerDecoderLayer(
            cfg, d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        self.decoder = AgentFormerDecoder(decoder_layer, num_decoder_layers)

        self.output_projection = nn.Linear(d_model, output_dim)

    def _build_src_key_padding_mask(
        self, target_padding_mask: torch.Tensor, h_in: int
    ) -> torch.Tensor:
        """
        Encoder source mask: marks zero-padded agent slots across all input timesteps.
        Derived from target_padding_mask by identifying agents where every target
        step is padding (i.e. the agent slot is entirely unused).

        Args:
            target_padding_mask: [B, H_out, N] — True = padding.
            h_in: number of encoder input timesteps.
        Returns:
            [B, H_in*N] — True = ignore (padded agent slot).
        """
        B, _, N = target_padding_mask.shape
        agent_padded = target_padding_mask.all(dim=1)  # [B, N]
        return agent_padded.unsqueeze(1).expand(-1, h_in, -1).reshape(B, h_in * N)

    def _build_tgt_key_padding_mask(
        self, target_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Decoder target mask: flat version of the full target padding mask.
        No permute needed — [B, H_out, N] layout already matches the
        interleaved sequence order (t=0,n=0..N-1), (t=1,n=0..N-1), ...

        Args:
            target_padding_mask: [B, H_out, N] — True = padding.
        Returns:
            [B, H_out*N] — True = ignore.
        """
        B, H, N = target_padding_mask.shape
        return target_padding_mask.reshape(B, H * N)

    def encode(
        self,
        input_traj: torch.Tensor,
        target_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_traj: [B, H_in, N, F] encoder input trajectory.
            target_padding_mask: [B, H_out, N] — True = padding. Used to derive
                src_key_padding_mask for zero-padded agent slots.

        Returns:
            [B, H_in, N, d_model] encoder output (memory).
        """
        B, H_in, N, _ = input_traj.shape

        input_emb = self.input_embedding(input_traj)
        # Flatten to [H_in*N, B, d_model] for transformer convention
        src = (
            input_emb.permute(1, 2, 0, 3)
            .reshape(H_in * N, B, self.d_model)
        )
        src = self.input_pos_encoding(src, num_a=N)

        src_key_padding_mask = None
        if target_padding_mask is not None:
            src_key_padding_mask = self._build_src_key_padding_mask(
                target_padding_mask, H_in
            )

        memory = self.encoder(
            src,
            mask=None,
            src_key_padding_mask=src_key_padding_mask,
            num_agent=self.num_agent,
        )

        # Reshape [H_in*N, B, d_model] -> [B, H_in, N, d_model]
        memory = memory.reshape(H_in, N, B, self.d_model).permute(2, 0, 1, 3)
        return memory

    def decode(
        self,
        dec_in_traj: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode the output trajectory.

        Args:
            dec_in_traj: Decoder input [B, H_out, N, F_dec].
            memory: Encoder memory [B, H_in, N, d_model].
            causal_mask: Block-causal mask [H_out*N, H_out*N] for autoregressive decoding.
            target_padding_mask: [B, H_out, N] — True = padding.

        Returns:
            Predicted trajectory [B, H_out, N, output_dim].
        """
        B, H_out, N, _ = dec_in_traj.shape
        H_in = memory.shape[1]

        dec_in_emb = self.dec_in_embedding(dec_in_traj)
        # Flatten to [H_out*N, B, d_model]
        tgt = (
            dec_in_emb.permute(1, 2, 0, 3)
            .reshape(H_out * N, B, self.d_model)
        )
        tgt = self.dec_in_pos_encoding(tgt, num_a=N)

        # Flatten memory to [H_in*N, B, d_model]
        memory_flat = (
            memory.permute(1, 2, 0, 3)
            .reshape(H_in * N, B, self.d_model)
        )

        tgt_key_padding_mask = None
        if target_padding_mask is not None:
            tgt_key_padding_mask = self._build_tgt_key_padding_mask(
                target_padding_mask
            )

        output, _ = self.decoder(
            tgt,
            memory_flat,
            tgt_mask=causal_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
            num_agent=self.num_agent,
            need_weights=False,
        )

        # Reshape [H_out*N, B, d_model] -> [B, H_out, N, output_dim]
        output = output.reshape(H_out, N, B, self.d_model).permute(2, 0, 1, 3)
        return self.output_projection(output)

    def forward(
        self,
        input_traj: torch.Tensor,
        dec_in_traj: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass through encoder and decoder.

        Args:
            input_traj: Input trajectory [B, H_in, N, encoder_input_dim].
            dec_in_traj: Decoder input [B, H_out, N, decoder_input_dim].
            causal_mask: Block-causal mask [H_out*N, H_out*N] for autoregressive decoding.
            target_padding_mask: [B, H_out, N] — True = padding. Passed to both
                encode (for src agent mask) and decode (for tgt mask).

        Returns:
            Predicted trajectory [B, H_out, N, output_dim].
        """
        memory = self.encode(input_traj, target_padding_mask=target_padding_mask)
        return self.decode(
            dec_in_traj,
            memory,
            causal_mask=causal_mask,
            target_padding_mask=target_padding_mask,
        )
