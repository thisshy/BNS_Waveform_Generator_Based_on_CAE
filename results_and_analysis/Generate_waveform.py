"""Utility helpers for generating BNS waveforms with the CAE models.

The original script hard coded dataset/model paths which made it difficult to
reuse in different environments.  The module now exposes a simple command line
interface so that the dataset location, model directory and sample index can be
configured without modifying the source code.  It also keeps the model loading
and scaling logic in testable helper functions so the module can be imported
from notebooks or other scripts.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Tuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _configure_gpu(cuda_visible_devices: str | None) -> None:
    """Configure TensorFlow GPU behaviour.

    Parameters
    ----------
    cuda_visible_devices:
        Optional CUDA device string (for example ``"0"``).  If ``None`` the
        environment variable is left untouched.
    """

    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print("❌ Failed to set memory growth:", exc)


def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the training set and return ``(X_train, y_amplitude, y_phase)``."""

    dataset_dict = np.load(dataset_path)
    try:
        X_train = dataset_dict["X_train"].astype("float32")
        y_amplitude = dataset_dict["y_amplitude"].astype("float32")
    except KeyError as exc:  # pragma: no cover - guard for user error only
        raise KeyError(
            "Dataset is missing the required keys 'X_train'/'y_amplitude'."
        ) from exc

    # Some datasets were stored with an extra trailing space in the key name.
    if "y_phase" in dataset_dict:
        phase_key = "y_phase"
    elif "y_phase " in dataset_dict:
        phase_key = "y_phase "
    else:  # pragma: no cover - guard for user error only
        raise KeyError("Dataset is missing the 'y_phase' array.")

    y_phase = dataset_dict[phase_key].astype("float32")
    return X_train, y_amplitude, y_phase


def build_scalers(
    X_train: np.ndarray, y_amplitude: np.ndarray, y_phase: np.ndarray
) -> Tuple[MinMaxScaler, StandardScaler, StandardScaler]:
    """Fit scalers for the dataset and return them."""

    scaler_X = MinMaxScaler().fit(X_train)
    scaler_y_phase = StandardScaler().fit(y_phase)
    scaler_y_amplitude = StandardScaler().fit(y_amplitude)
    return scaler_X, scaler_y_amplitude, scaler_y_phase


def load_models(model_directory: str) -> Tuple[tf.keras.Model, ...]:
    """Load the four CAE models from ``model_directory``."""

    def _load(name: str) -> tf.keras.Model:
        path = os.path.join(model_directory, name)
        if not os.path.exists(path):  # pragma: no cover - guard for user error
            raise FileNotFoundError(f"Model '{name}' was not found in '{path}'.")
        return tf.keras.models.load_model(path)

    return (
        _load("cAE_encoder_phase_conditional"),
        _load("cAE_decoder_phase"),
        _load("cAE_encoder_amplitude_conditional"),
        _load("cAE_decoder_amplitude"),
    )


def predict_waveform_from_sample(
    x_sample: Iterable[float],
    scaler_X: MinMaxScaler,
    scaler_y_amplitude: StandardScaler,
    scaler_y_phase: StandardScaler,
    encoder_phase: tf.keras.Model,
    decoder_phase: tf.keras.Model,
    encoder_amplitude: tf.keras.Model,
    decoder_amplitude: tf.keras.Model,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict a waveform from a single parameter sample."""

    x_sample = np.array(x_sample, dtype="float32").reshape(1, -1)
    x_scaled = scaler_X.transform(x_sample)

    phase_latent = encoder_phase(x_scaled)
    phase_AI = decoder_phase(phase_latent).numpy().ravel()

    amplitude_latent = encoder_amplitude(x_scaled)
    amplitude_AI = decoder_amplitude(amplitude_latent).numpy().ravel()

    phase_AI = scaler_y_phase.inverse_transform(phase_AI.reshape(1, -1)).ravel()
    amplitude_AI = scaler_y_amplitude.inverse_transform(
        amplitude_AI.reshape(1, -1)
    ).ravel()

    return cal_hphc_from_amp_ph(amplitude_AI, phase_AI)


def cal_hphc_from_amp_ph(
    amplitude: Iterable[float], phase: Iterable[float]
) -> Tuple[np.ndarray, np.ndarray]:
    amplitude = np.asarray(amplitude)
    phase = np.asarray(phase)
    hp = amplitude * np.cos(phase)
    hc = amplitude * np.sin(phase)
    return hp, hc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a BNS waveform.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the training dataset (NPZ) used to fit the scalers.",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Directory containing the trained CAE models.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the sample in the dataset used to generate the waveform.",
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help="CUDA_VISIBLE_DEVICES value (set to '' to force CPU).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_gpu(args.cuda_devices)

    X_train, y_amplitude, y_phase = load_dataset(args.dataset)
    sample = X_train[args.sample_index]

    scaler_X, scaler_y_amplitude, scaler_y_phase = build_scalers(
        X_train, y_amplitude, y_phase
    )

    (
        encoder_phase,
        decoder_phase,
        encoder_amplitude,
        decoder_amplitude,
    ) = load_models(args.models)

    hp_pred, hc_pred = predict_waveform_from_sample(
        sample,
        scaler_X,
        scaler_y_amplitude,
        scaler_y_phase,
        encoder_phase,
        decoder_phase,
        encoder_amplitude,
        decoder_amplitude,
    )

    print("hp:", hp_pred)
    print("hc:", hc_pred)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
















