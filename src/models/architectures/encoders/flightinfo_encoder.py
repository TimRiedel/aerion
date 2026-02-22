import torch
import torch.nn as nn


class FlightInfoEncoder(nn.Module):
    def __init__(self, d_model: int, input_dim: int):
        """
        Args:
            d_model: Model dimension (must match the transformer's d_model)
            input_dim: Number of input features
        """
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, d_model)
    
    def forward(self, flightinfo: torch.Tensor) -> torch.Tensor:
        """
        Input: Flight info tensor [batch, input_dim]
        Output: Encoded flight info [batch, 1, d_model] for cross-attention
        """
        # Project to d_model and add sequence dimension for cross-attention
        encoded = self.projection(flightinfo)  # [batch, d_model]
        return encoded.unsqueeze(1)  # [batch, 1, d_model]