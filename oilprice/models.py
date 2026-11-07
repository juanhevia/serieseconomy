"""Tres redes de una capa recurrente con salida de un precio escalado."""

import torch
from torch import nn


class RecurrentRegressor(nn.Module):
    def __init__(self, architecture, hidden_size, input_size=1):
        super().__init__()
        layers = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
        if architecture not in layers or hidden_size < 1 or input_size < 1:
            raise ValueError("Arquitectura o dimensiones no válidas.")
        self.recurrent = layers[architecture](input_size=input_size, hidden_size=hidden_size,
                                              num_layers=1, batch_first=True, bidirectional=False)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence, _ = self.recurrent(x)
        return self.output(sequence[:, -1, :])


def load_regressor(weights_path, metadata, device="cpu"):
    """Carga pesos, no objetos Python arbitrarios. Solo usar artefactos propios."""
    model = RecurrentRegressor(metadata["architecture"], metadata["candidate"]["hidden_size"],
                               metadata["input_size"])
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model.to(device).eval()
