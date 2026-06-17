from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from all_algorithms_runner.common import Result, ratio_series, robust_threshold


SEQ_LENGTH = 8


class PatchTST(nn.Module):
    def __init__(self, input_size: int, patch_size: int, hidden_size: int, num_layers: int, output_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Linear(input_size * patch_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unfold(1, self.patch_size, self.patch_size).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return self.fc(x.mean(dim=1))


def _sequences(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= SEQ_LENGTH:
        return np.array([]), np.array([])
    x = np.array([values[idx : idx + SEQ_LENGTH] for idx in range(len(values) - SEQ_LENGTH)])
    y = np.array(values[SEQ_LENGTH:])
    return x, y


def load_model(model_dir: Path) -> tuple[PatchTST, object, object]:
    model = PatchTST(
        input_size=1,
        patch_size=2,
        hidden_size=64,
        num_layers=2,
        output_size=1,
        num_heads=4,
    )
    model.load_state_dict(torch.load(model_dir / "patchTST_model.pth", map_location="cpu", weights_only=True))
    model.eval()
    return (
        model,
        joblib.load(model_dir / "patchTST_scaler_X.pkl"),
        joblib.load(model_dir / "patchTST_scaler_y.pkl"),
    )


def run(df: pd.DataFrame, day: str, rg_id: str, prefix: str, model_bundle=None) -> list[Result]:
    if model_bundle is None:
        raise ValueError("patchTST requires a loaded model bundle")
    model, scaler_x, scaler_y = model_bundle
    rows = []
    for serial, serial_df in df.groupby(level="serialno"):
        values = ratio_series(serial_df).to_numpy()
        x_test, y_test = _sequences(values)
        if x_test.size == 0:
            continue
        x_scaled = scaler_x.transform(x_test)
        tensor = torch.tensor(x_scaled, dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model(tensor).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_scaled)
        rows.append((str(serial), mean_squared_error(y_test, pred)))

    threshold = robust_threshold([mse for _, mse in rows], sigma=2.0)
    return [
        Result(prefix, day, "patchTST", rg_id, serial)
        for serial, mse in rows
        if mse > threshold
    ]
