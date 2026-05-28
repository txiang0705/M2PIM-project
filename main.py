#!/usr/bin/env python3
"""
M2PIM personalized-calibration example for manuscript code sharing.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


# =============================================================================
# User-adjustable settings
# =============================================================================

# Input/output paths. Relative paths are resolved from this script folder.
DATA_CSV = Path("data/demo_subject_exercise_recovery.csv")
OUTPUT_DIR = Path("results")
TARGETS = ["SBP", "DBP"]

# Training settings used in the manuscript.
MAX_EPOCHS = 5000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 100
RANDOM_SEED = 0
VERBOSE = False

# Composite M2PIM objective weights from the manuscript/SI.
# alpha_taylor weights the Taylor temporal physics loss, which encourages local
# beat-to-beat BP changes to agree with a first-order expansion over physiology.
ALPHA_TAYLOR = 10.0

# beta_hemodynamic weights the pressure-flow-resistance derivative loss, which
# aligns model-implied dBP/dt with the derivative derived from CO/R proxies.
BETA_HEMODYNAMIC = 1.0

# gamma_cor weights the physiological consistency loss, which regularizes the
# learned CO-like and R-like latent states toward independently computed proxies.
GAMMA_COR = 0.01


# Four synchronized beat-level waveform channels used by the CNN-LSTM backbone.
WAVEFORM_COLUMNS = ["ECG_Waveform", "PPG_Waveform", "IPG_Waveform", "Temp_Waveform"]

# The 25 non-waveform variables described in the manuscript and SI:
# one protocol/stage variable, 19 physiological features, and five demographics.
NON_WAVEFORM_FEATURES = [
    "Stage",
    "HR_ECG",
    "R_peak",
    "PTT_ECG_IPG",
    "HR_IPGmax",
    "HR_IPGmin",
    "IPGmax_Zmax",
    "IPGmin_Zmin",
    "Delta_IPG_dZ",
    "IPG_area_Zarea",
    "IIR",
    "PTT_ECG_PPG",
    "HR_PPG",
    "PPGmax",
    "PPGmin",
    "Delta_PPG_PA",
    "PPG_width",
    "PPG_area_PArea",
    "PIR",
    "Ave_ST",
    "sex",
    "age",
    "height_cm",
    "weight_kg",
    "bmi",
]

# Eight physiologically interpretable variables used in the soft constraints.
PHYSIOLOGY_FEATURES = [
    "HR_ECG",
    "Delta_PPG_PA",
    "PPG_area_PArea",
    "PTT_ECG_PPG",
    "IPGmax_Zmax",
    "IPG_area_Zarea",
    "Delta_IPG_dZ",
    "Ave_ST",
]
PHYSIOLOGY_FEATURE_INDICES = [NON_WAVEFORM_FEATURES.index(name) for name in PHYSIOLOGY_FEATURES]

# Train/test split protocol for personalized calibration.  The sparse labelled
# calibration set contains one beat from each 1 mmHg BP bin; all remaining beats
# are held out for final testing.
CALIBRATION_BIN_WIDTH_MMHG = 1.0
CALIBRATION_SAMPLES_PER_BIN = 1
VALIDATION_FRACTION_OF_CALIBRATION = 0.20


def parse_waveform(value: object) -> np.ndarray:
    """Parse a waveform stored as a bracketed string in the demo CSV."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    text = str(value).strip().replace("\n", " ")
    text = text.strip("[]")
    arr = np.fromstring(text, sep=" ", dtype=np.float32)
    if arr.size == 0:
        arr = np.fromstring(text.replace(",", " "), sep=" ", dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Could not parse waveform cell.")
    return arr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def second_order_rnn_context(device: torch.device):

    if device.type == "cuda":
        return torch.backends.cudnn.flags(enabled=False)
    return nullcontext()


def personalized_calibration_split(
    df: pd.DataFrame,
    target: str,
    seed: int,
    bin_width_mmhg: float = 1.0,
    samples_per_bin: int = 1,
    min_train_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray, int]:

    bp = df[target].to_numpy(dtype=float)
    segment_ids = np.floor(bp / bin_width_mmhg).astype(int)
    rng = np.random.default_rng(seed)

    calibration_indices: list[int] = []
    for segment_id in sorted(np.unique(segment_ids)):
        candidates = np.where(segment_ids == segment_id)[0]
        n_select = min(samples_per_bin, len(candidates))
        calibration_indices.extend(rng.choice(candidates, size=n_select, replace=False).tolist())

    calibration_indices = sorted(set(calibration_indices))
    if len(calibration_indices) < min_train_samples:
        remaining = np.setdiff1d(np.arange(len(df)), np.asarray(calibration_indices, dtype=int))
        needed = min(min_train_samples - len(calibration_indices), len(remaining))
        if needed > 0:
            calibration_indices.extend(rng.choice(remaining, size=needed, replace=False).tolist())

    calibration_indices = np.asarray(sorted(set(calibration_indices)), dtype=int)
    mask = np.zeros(len(df), dtype=bool)
    mask[calibration_indices] = True
    test_indices = np.where(~mask)[0]
    if len(test_indices) == 0:
        raise ValueError("Calibration split produced no held-out test beats.")
    return calibration_indices, test_indices, len(np.unique(segment_ids))


def split_calibration_train_validation(
    calibration_indices: np.ndarray,
    seed: int,
    validation_fraction: float = VALIDATION_FRACTION_OF_CALIBRATION,
    min_train_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray]:

    if len(calibration_indices) <= min_train_samples:
        raise ValueError("Not enough calibration beats to create a validation split.")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(calibration_indices, dtype=int).copy()
    rng.shuffle(shuffled)

    n_validation = int(round(len(shuffled) * validation_fraction))
    n_validation = max(1, n_validation)
    n_validation = min(n_validation, len(shuffled) - min_train_samples)

    validation_indices = np.sort(shuffled[:n_validation])
    train_indices = np.sort(shuffled[n_validation:])
    return train_indices, validation_indices


class PreparedData:
    def __init__(
        self,
        train_waveforms: torch.Tensor,
        val_waveforms: torch.Tensor,
        test_waveforms: torch.Tensor,
        all_waveforms: torch.Tensor,
        train_features: torch.Tensor,
        val_features: torch.Tensor,
        test_features: torch.Tensor,
        all_features: torch.Tensor,
        y_train: torch.Tensor,
        y_val: torch.Tensor,
        y_test: torch.Tensor,
        y_test_mmHg: np.ndarray,
        target_scaler: StandardScaler,
        test_indices: np.ndarray,
    ):
        self.train_waveforms = train_waveforms
        self.val_waveforms = val_waveforms
        self.test_waveforms = test_waveforms
        self.all_waveforms = all_waveforms
        self.train_features = train_features
        self.val_features = val_features
        self.test_features = test_features
        self.all_features = all_features
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.y_test_mmHg = y_test_mmHg
        self.target_scaler = target_scaler
        self.test_indices = test_indices


def fit_channelwise_scaler(train_waveforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit simple per-channel waveform normalization using calibration beats."""
    means = train_waveforms.mean(axis=(0, 2), keepdims=True)
    stds = train_waveforms.std(axis=(0, 2), keepdims=True)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return means, stds


def prepare_data(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    target: str,
) -> PreparedData:

    for col in WAVEFORM_COLUMNS:
        df[col] = df[col].apply(parse_waveform)

    lengths = {len(arr) for col in WAVEFORM_COLUMNS for arr in df[col]}
    if len(lengths) != 1:
        raise ValueError(f"Waveform lengths are inconsistent: {sorted(lengths)}")

    waveforms = np.stack(
        [np.stack(df[col].to_numpy(), axis=0) for col in WAVEFORM_COLUMNS],
        axis=1,
    ).astype(np.float32)
    features = df[NON_WAVEFORM_FEATURES].to_numpy(dtype=np.float32)


    train_wave = waveforms[train_idx]
    val_wave = waveforms[val_idx]
    test_wave = waveforms[test_idx]
    means, stds = fit_channelwise_scaler(train_wave)
    train_wave = (train_wave - means) / stds
    val_wave = (val_wave - means) / stds
    test_wave = (test_wave - means) / stds
    all_wave = (waveforms - means) / stds

    feature_scaler = StandardScaler().fit(features[train_idx])
    train_features = feature_scaler.transform(features[train_idx]).astype(np.float32)
    val_features = feature_scaler.transform(features[val_idx]).astype(np.float32)
    test_features = feature_scaler.transform(features[test_idx]).astype(np.float32)
    all_features = feature_scaler.transform(features).astype(np.float32)

    target_scaler = StandardScaler().fit(df.loc[train_idx, [target]].to_numpy(dtype=np.float32))
    y_train = target_scaler.transform(df.loc[train_idx, [target]].to_numpy(dtype=np.float32)).astype(np.float32)
    y_val = target_scaler.transform(df.loc[val_idx, [target]].to_numpy(dtype=np.float32)).astype(np.float32)
    y_test = target_scaler.transform(df.loc[test_idx, [target]].to_numpy(dtype=np.float32)).astype(np.float32)

    return PreparedData(
        train_waveforms=torch.tensor(train_wave, dtype=torch.float32),
        val_waveforms=torch.tensor(val_wave, dtype=torch.float32),
        test_waveforms=torch.tensor(test_wave, dtype=torch.float32),
        all_waveforms=torch.tensor(all_wave, dtype=torch.float32),
        train_features=torch.tensor(train_features, dtype=torch.float32),
        val_features=torch.tensor(val_features, dtype=torch.float32),
        test_features=torch.tensor(test_features, dtype=torch.float32),
        all_features=torch.tensor(all_features, dtype=torch.float32),
        y_train=torch.tensor(y_train, dtype=torch.float32),
        y_val=torch.tensor(y_val, dtype=torch.float32),
        y_test=torch.tensor(y_test, dtype=torch.float32),
        y_test_mmHg=df.loc[test_idx, target].to_numpy(dtype=np.float32),
        target_scaler=target_scaler,
        test_indices=test_idx,
    )


class M2PIMCNNLSTM(nn.Module):
    """
    The CNN branch extracts local waveform morphology, the LSTM branch captures
    within-beat temporal structure, and the latent heads estimate CO-like and
    R-like physiological states used by the constraint loss.
    """

    def __init__(self, n_waveform_channels: int = 4, n_scalar_features: int = 25, hidden: int = 64):
        super().__init__()
        self.n_waveform_channels = n_waveform_channels
        self.n_scalar_features = n_scalar_features
        self.n_input_channels = n_waveform_channels + n_scalar_features

        self.conv1 = nn.Conv1d(self.n_input_channels, hidden, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.temporal_lstm = nn.LSTM(
            input_size=self.n_input_channels,
            hidden_size=hidden,
            num_layers=4,
            batch_first=True,
        )
        fused_dim = hidden + hidden + n_scalar_features + (2 * n_waveform_channels)
        self.fusion_lstm = nn.LSTM(input_size=fused_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.representation = nn.Linear(hidden, 60)

        self.direct_head = nn.Linear(60, 1)
        self.co_head = nn.Linear(60, 1)
        self.r_head = nn.Linear(60, 1)
        self.phys_bp_head = nn.Linear(3, 1)

        self.a1 = nn.Parameter(torch.tensor(1.0))
        self.a2 = nn.Parameter(torch.tensor(1.0))
        self.a3 = nn.Parameter(torch.tensor(1.0))
        self.b1 = nn.Parameter(torch.tensor(1.0))
        self.b2 = nn.Parameter(torch.tensor(1.0))
        self.b3 = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.r0 = nn.Parameter(torch.tensor(15.0))

    def make_sequence(self, waveforms: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        repeated_features = scalar_features.unsqueeze(-1).expand(-1, -1, waveforms.shape[-1])
        return torch.cat([waveforms, repeated_features], dim=1)

    def forward(self, waveforms: torch.Tensor, scalar_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_channels = self.make_sequence(waveforms, scalar_features)

        # CNN morphology encoder.
        cnn = F.relu(self.conv1(sequence_channels))
        cnn = self.pool(cnn)
        cnn = F.relu(self.conv2(cnn))
        cnn = self.pool(cnn)
        cnn_repr = F.adaptive_avg_pool1d(cnn, 1).squeeze(-1)

        # LSTM temporal encoder over the samples within each beat.
        sequence_steps = sequence_channels.transpose(1, 2)
        _, (hidden_state, _) = self.temporal_lstm(sequence_steps)
        lstm_repr = hidden_state[-1]

        # A lightweight global waveform descriptor helps the demo remain stable.
        waveform_mean = waveforms.mean(dim=-1)
        waveform_std = waveforms.std(dim=-1)
        fused = torch.cat([cnn_repr, lstm_repr, scalar_features, waveform_mean, waveform_std], dim=1)
        fused_seq = fused.unsqueeze(1)
        fusion_out, _ = self.fusion_lstm(fused_seq)
        rep = F.relu(self.representation(fusion_out[:, -1, :]))

        bp_direct = self.direct_head(rep)
        co_state = F.softplus(self.co_head(rep))
        r_state = F.softplus(self.r_head(rep))
        bp_phys = self.phys_bp_head(torch.cat([co_state, r_state, co_state * r_state], dim=1))

        # Convex fusion follows the manuscript description: mostly neural BP,
        # with a smaller contribution from the CO/R-derived BP component.
        bp_pred = 0.8 * bp_direct + 0.2 * bp_phys
        return bp_pred, co_state, r_state

    @staticmethod
    def stable_standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        std = x.std()
        if torch.isfinite(std) and std > eps:
            return (x - x.mean()) / std
        return x - x.mean()

    @staticmethod
    def clamp_away_from_zero(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        return torch.where(torch.abs(x) < eps, sign * eps, x)

    def hemodynamic_proxies(self, phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build CO-like and R-like proxy targets from measurable features."""
        hr, pa, parea, ptt, zmax, zarea, dz, temp = [phys[:, i : i + 1] for i in range(8)]
        safe_ptt = self.clamp_away_from_zero(ptt)
        stroke_volume_proxy = self.a1 * pa + self.a2 * parea + self.a3 / safe_ptt
        co_proxy = stroke_volume_proxy * hr
        resistance_proxy = (
            self.r0 * torch.exp(torch.clamp(-self.gamma * temp, -20.0, 20.0))
            + self.b1 * zmax.pow(2)
            + self.b2 * zarea
            + self.b3 * dz
        )
        return co_proxy, resistance_proxy

    def physics_loss(
        self,
        pred_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        y_true: torch.Tensor,
        physics_pred_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        physics_scalar_features: torch.Tensor,
        alpha_taylor: float,
        beta_hemodynamic: float,
        gamma_cor: float,
    ) -> torch.Tensor:

        y_pred, _, _ = pred_tuple
        bp_regression_loss = torch.mean((y_pred - y_true) ** 2)
        physics_pred, physics_co_state, physics_r_state = physics_pred_tuple

        # L_Taylor: predicted beat-to-beat changes should match the local
        # first-order Taylor expansion implied by physiological feature changes.
        phys = physics_scalar_features[:, PHYSIOLOGY_FEATURE_INDICES]
        grads_all = torch.autograd.grad(
            physics_pred,
            physics_scalar_features,
            grad_outputs=torch.ones_like(physics_pred),
            retain_graph=True,
            create_graph=True,
        )[0]
        grads = grads_all[:, PHYSIOLOGY_FEATURE_INDICES]
        taylor_next = physics_pred[:-1, 0] + torch.sum(grads[:-1] * (phys[1:] - phys[:-1]), dim=1)
        taylor_physics_loss = torch.mean((taylor_next - physics_pred[1:, 0]) ** 2)

        # L_Hemo: simplified pressure-flow-resistance derivative consistency.
        # This is a soft guide rather than a hard hemodynamic simulator.
        co_proxy, r_proxy = self.hemodynamic_proxies(phys)
        dco = torch.diff(co_proxy.squeeze(), dim=0)
        dr = torch.diff(r_proxy.squeeze(), dim=0)
        d_bp_physics = r_proxy[:-1, 0] * dco + co_proxy[:-1, 0] * dr
        d_bp_physics = self.stable_standardize(d_bp_physics)
        d_bp_model = torch.diff(physics_pred.squeeze(), dim=0)
        hemodynamic_derivative_loss = torch.mean((d_bp_model - d_bp_physics) ** 2)

        # L_CO/R: keeps learned CO-like and R-like states aligned with
        # independently derived physiological proxy estimates.
        co_consistency_loss = torch.mean(
            (self.stable_standardize(physics_co_state[:-1, 0]) - self.stable_standardize(co_proxy[:-1, 0])) ** 2
        )
        r_consistency_loss = torch.mean(
            (self.stable_standardize(physics_r_state[:-1, 0]) - self.stable_standardize(r_proxy[:-1, 0])) ** 2
        )
        cor_consistency_loss = co_consistency_loss + r_consistency_loss

        return (
            bp_regression_loss
            + alpha_taylor * taylor_physics_loss
            + beta_hemodynamic * hemodynamic_derivative_loss
            + gamma_cor * cor_consistency_loss
        )


def inverse_transform(scaler: StandardScaler, y_scaled: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)


def calculate_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - reference
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "ME": float(np.mean(error)),
        "SD_error": float(np.std(error, ddof=1)),
        "within_5mmHg_percent": float(np.mean(np.abs(error) <= 5.0) * 100.0),
        "within_10mmHg_percent": float(np.mean(np.abs(error) <= 10.0) * 100.0),
        "within_15mmHg_percent": float(np.mean(np.abs(error) <= 15.0) * 100.0),
    }


def stable_zscore_np(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a numerically stable z-score for one-dimensional arrays."""
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    if std < eps:
        return values - values.mean()
    return (values - values.mean()) / std


def physiology_proxy_scores(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute transparent CO-like and R-like proxy scores for orientation.
    """
    co_proxy = (
        stable_zscore_np(df["HR_ECG"].to_numpy())
        + stable_zscore_np(df["Delta_PPG_PA"].to_numpy())
        + stable_zscore_np(df["PPG_area_PArea"].to_numpy())
        - stable_zscore_np(df["PTT_ECG_PPG"].to_numpy())
    )
    r_proxy = (
        stable_zscore_np(df["IPGmax_Zmax"].to_numpy())
        + stable_zscore_np(df["IPG_area_Zarea"].to_numpy())
        + stable_zscore_np(df["Delta_IPG_dZ"].to_numpy())
        - stable_zscore_np(df["Ave_ST"].to_numpy())
    )
    return co_proxy, r_proxy


def orient_latent_state(all_state: np.ndarray, test_state: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    """Flip a latent state only when its arbitrary sign opposes its proxy."""
    all_state = np.asarray(all_state, dtype=np.float64).reshape(-1)
    test_state = np.asarray(test_state, dtype=np.float64).reshape(-1)
    proxy = np.asarray(proxy, dtype=np.float64).reshape(-1)
    corr = np.corrcoef(stable_zscore_np(all_state), stable_zscore_np(proxy))[0, 1]
    if np.isfinite(corr) and corr < 0:
        return -test_state
    return test_state


def train_one_target(df: pd.DataFrame, args: argparse.Namespace, target: str, device: torch.device) -> dict[str, object]:
    """Train one personalized M2PIM model for SBP or DBP and evaluate it."""
    split_seed = args.seed + (1000 if target == "DBP" else 0)
    calibration_idx, test_idx, _ = personalized_calibration_split(
        df,
        target=target,
        seed=split_seed,
        bin_width_mmhg=args.calibration_bin_width,
        samples_per_bin=args.calibration_samples_per_bin,
    )
    train_idx, val_idx = split_calibration_train_validation(
        calibration_idx,
        seed=split_seed + 17,
        validation_fraction=args.validation_fraction,
    )
    data = prepare_data(df.copy(), train_idx, val_idx, test_idx, target)

    train_wave = data.train_waveforms.to(device)
    train_features_base = data.train_features.to(device)
    val_wave = data.val_waveforms.to(device)
    val_features = data.val_features.to(device)
    all_wave = data.all_waveforms.to(device)
    all_features_base = data.all_features.to(device)
    y_train = data.y_train.to(device)
    y_val = data.y_val.to(device)

    set_seed(args.seed + (23 if target == "SBP" else 1023))
    model = M2PIMCNNLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if len(train_wave) > args.batch_size:
            batch_indices = torch.randperm(len(train_wave), device=device)[: args.batch_size]
            train_wave_batch = train_wave[batch_indices]
            train_features_batch_base = train_features_base[batch_indices]
            y_train_batch = y_train[batch_indices]
        else:
            train_wave_batch = train_wave
            train_features_batch_base = train_features_base
            y_train_batch = y_train

        train_features = train_features_batch_base.clone().detach().requires_grad_(True)
        all_features = all_features_base.clone().detach().requires_grad_(True)
        with second_order_rnn_context(device):
            pred_tuple = model(train_wave_batch, train_features)
            all_pred_tuple = model(all_wave, all_features)
            loss = model.physics_loss(
                pred_tuple,
                y_train_batch,
                physics_pred_tuple=all_pred_tuple,
                physics_scalar_features=all_features,
                alpha_taylor=args.alpha_taylor,
                beta_hemodynamic=args.beta_hemodynamic,
                gamma_cor=args.gamma_cor,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred, _, _ = model(val_wave, val_features)
            validation_loss = torch.mean((val_pred - y_val) ** 2).item()

        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1

        if args.verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(
                f"{target} M2PIM epoch {epoch + 1:4d}/{args.epochs}: "
                f"train loss={loss.item():.4f}, validation loss={validation_loss:.4f}"
            )

        if epochs_without_improvement >= args.early_stopping_patience:
            if args.verbose:
                print(
                    f"{target} early stopping at epoch {epoch + 1}; "
                    f"best validation loss={best_validation_loss:.4f} at epoch {best_epoch}."
                )
            break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_scaled, co_state, r_state = model(data.test_waveforms.to(device), data.test_features.to(device))
        _, all_co_state, all_r_state = model(data.all_waveforms.to(device), data.all_features.to(device))
    pred_mmHg = inverse_transform(data.target_scaler, pred_scaled.cpu().numpy().reshape(-1))
    ref_mmHg = data.y_test_mmHg
    co_proxy, r_proxy = physiology_proxy_scores(df)
    co_state_out = orient_latent_state(
        all_co_state.cpu().numpy(),
        co_state.cpu().numpy(),
        co_proxy,
    )
    r_state_out = orient_latent_state(
        all_r_state.cpu().numpy(),
        r_state.cpu().numpy(),
        r_proxy,
    )

    phase = (
        df.loc[test_idx, "phase"].to_numpy()
        if "phase" in df.columns
        else df.loc[test_idx, "Stage"].astype(str).to_numpy()
    )
    predictions = pd.DataFrame(
        {
            "beat_index": data.test_indices,
            "phase": phase,
            "target": target,
            "reference_mmHg": ref_mmHg,
            "m2pim_prediction_mmHg": pred_mmHg,
            "CO_like_state": co_state_out.reshape(-1),
            "R_like_state": r_state_out.reshape(-1),
        }
    )

    metrics = calculate_metrics(
        predictions["reference_mmHg"].to_numpy(),
        predictions["m2pim_prediction_mmHg"].to_numpy(),
    )

    return {
        "target": target,
        "metrics": metrics,
        "predictions": predictions,
    }


def plot_predictions(predictions: pd.DataFrame, output_path: Path, target: str) -> None:
    """Save a publication-style trace plot for the held-out demo beats."""
    x = np.arange(len(predictions))
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    ax.plot(
        x,
        predictions["reference_mmHg"],
        color="black",
        linestyle="--",
        linewidth=3.0,
        label=f"Reference {target}",
        zorder=3,
    )
    ax.scatter(
        x,
        predictions["m2pim_prediction_mmHg"],
        color="#E69F00",
        s=32,
        label="MPIM Estimation",
        zorder=5,
    )

    if "phase" in predictions.columns:
        phases = predictions["phase"].astype(str).to_numpy()
        run_start = 0
        for idx in range(1, len(phases) + 1):
            if idx == len(phases) or phases[idx] != phases[run_start]:
                if run_start > 0:
                    ax.axvline(run_start - 0.5, color="#555555", linestyle="--", linewidth=1.2, alpha=0.65)
                run_start = idx

    ax.set_title(f"{target} Estimation for Demo Subject", fontsize=18)
    ax.set_xlabel("Beats")
    ax.set_ylabel(f"{target} (mmHg)")
    ax.grid(True, color="#B8B8B8", linewidth=1.0, alpha=0.75)
    ax.legend(loc="lower left", fontsize=12, frameon=True, facecolor="white", edgecolor="#CCCCCC")
    ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Optional command-line overrides for the user-adjustable settings above."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--targets", nargs="+", default=TARGETS, choices=["SBP", "DBP"])
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--calibration-bin-width", type=float, default=CALIBRATION_BIN_WIDTH_MMHG)
    parser.add_argument("--calibration-samples-per-bin", type=int, default=CALIBRATION_SAMPLES_PER_BIN)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION_OF_CALIBRATION)
    parser.add_argument("--alpha-taylor", type=float, default=ALPHA_TAYLOR)
    parser.add_argument("--beta-hemodynamic", type=float, default=BETA_HEMODYNAMIC)
    parser.add_argument("--gamma-cor", type=float, default=GAMMA_COR)
    parser.add_argument("--verbose", action="store_true", default=VERBOSE)
    return parser.parse_args()


def main() -> None:
    """Entry point: load demo data, train target-specific models, save outputs."""
    args = parse_args()
    set_seed(args.seed)
    script_dir = Path(__file__).resolve().parent
    data_path = args.data if args.data.is_absolute() else script_dir / args.data
    output_dir = args.output_dir if args.output_dir.is_absolute() else script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    required = set(["SBP", "DBP", *WAVEFORM_COLUMNS, *NON_WAVEFORM_FEATURES])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Input: {data_path}")

    summaries = []
    for target in args.targets:
        result = train_one_target(df, args, target, device)
        predictions = result.pop("predictions")
        pred_path = output_dir / f"demo_{target.lower()}_predictions.csv"
        predictions.to_csv(pred_path, index=False)
        plot_predictions(predictions, output_dir / f"demo_{target.lower()}_trace.png", target)

        flat = {
            "target": result["target"],
            **{f"m2pim_{k}": v for k, v in result["metrics"].items()},
        }
        summaries.append(flat)
        print(f"{target}: M2PIM MAE={flat['m2pim_MAE']:.2f} mmHg")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "demo_metrics_summary.csv", index=False)
    with (output_dir / "demo_metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()
