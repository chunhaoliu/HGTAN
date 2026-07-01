"""Classical threat-assessment baselines for ATUAV.

The neural models are evaluated against several domain-standard multi-criteria
decision-making methods. These methods are intentionally simple, transparent,
and fast, making them useful references for TAES-style threat-assessment
experiments.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler

from utils.config import (
    ALL_FEATURES,
    FEATURE_RISK_DIRECTION,
    N_CLASSES,
    N_URGENCY,
    PRIOR_WEIGHTS_TENSOR,
)

TRADITIONAL_MODEL_NAMES = (
    "TOPSIS",
    "GRA",
    "Fuzzy",
    "Entropy-TOPSIS",
    "Combined-TOPSIS",
    "TemporalHMM",
)

URGENCY_FEATURE_NAMES = (
    "mission_type",
    "distance",
    "velocity",
    "time_to_arrival",
    "swarm_size",
    "defense_capability",
    "target_asset_value",
    "track_confidence",
)
URGENCY_FEATURE_INDICES = tuple(ALL_FEATURES.index(name) for name in URGENCY_FEATURE_NAMES)


def entropy_weight(x: np.ndarray) -> np.ndarray:
    """Compute objective feature weights with the entropy-weight method."""
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-10, None)
    n_samples = x.shape[0]
    probabilities = x / (x.sum(axis=0, keepdims=True) + 1e-10)
    probabilities = np.clip(probabilities, 1e-10, 1.0)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=0) / np.log(n_samples + 1e-10)
    diversity = 1.0 - entropy
    return diversity / (diversity.sum() + 1e-10)


def critic_weight(x: np.ndarray) -> np.ndarray:
    """Compute objective feature weights with the CRITIC method."""
    x = np.asarray(x, dtype=np.float64)
    std = x.std(axis=0) + 1e-10
    corr = np.nan_to_num(np.corrcoef(x.T), nan=0.0)
    conflict = np.sum(1.0 - np.abs(corr), axis=1)
    information = std * conflict
    return information / (information.sum() + 1e-10)


def combination_weight(
    x: np.ndarray,
    prior_weights: np.ndarray | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend expert prior weights and entropy weights."""
    objective_weights = entropy_weight(x)
    if prior_weights is None:
        prior_weights = np.asarray(PRIOR_WEIGHTS_TENSOR, dtype=np.float64)
    combined = alpha * prior_weights + (1.0 - alpha) * objective_weights
    return combined / (combined.sum() + 1e-10)


def feature_names_for_indices(feature_indices: tuple[int, ...] | None) -> list[str]:
    if feature_indices is None:
        return list(ALL_FEATURES)
    return [ALL_FEATURES[index] for index in feature_indices]


def select_features(x: np.ndarray, feature_indices: tuple[int, ...] | None) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if feature_indices is None:
        return values
    return values[:, feature_indices]


def prior_weights_for_indices(feature_indices: tuple[int, ...] | None) -> np.ndarray:
    if feature_indices is None:
        weights = np.asarray(PRIOR_WEIGHTS_TENSOR, dtype=np.float64)
    else:
        weights = np.asarray(PRIOR_WEIGHTS_TENSOR, dtype=np.float64)[list(feature_indices)]
    return weights / (weights.sum() + 1e-10)


def orient_features_for_risk(x_scaled: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
    """Convert mixed-direction indicators into a higher-is-riskier convention."""
    oriented = np.asarray(x_scaled, dtype=np.float64).copy()
    names = feature_names or list(ALL_FEATURES)
    for feature_idx, feature_name in enumerate(names):
        if FEATURE_RISK_DIRECTION.get(feature_name, "high") == "low":
            oriented[:, feature_idx] = 1.0 - oriented[:, feature_idx]
    return np.clip(oriented, 0.0, 1.0)


class TOPSISClassifier(BaseEstimator, ClassifierMixin):
    """Technique for order preference by similarity to ideal solution."""

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        use_entropy_weight: bool = False,
        feature_indices: tuple[int, ...] | None = None,
    ):
        self.n_classes = n_classes
        self.use_entropy_weight = use_entropy_weight
        self.feature_indices = feature_indices
        self.weights: np.ndarray | None = None
        self.thresholds: list[float] | None = None
        self.scaler = MinMaxScaler()
        self.feature_names = feature_names_for_indices(feature_indices)

    def fit(self, x, y=None):
        del y
        x_selected = select_features(x, self.feature_indices)
        x_scaled = orient_features_for_risk(self.scaler.fit_transform(x_selected), self.feature_names)
        self.weights = entropy_weight(x_scaled) if self.use_entropy_weight else prior_weights_for_indices(self.feature_indices)
        scores = self._compute_scores(x_scaled)
        self.thresholds = _percentile_thresholds(scores, self.n_classes)
        return self

    def predict(self, x):
        x_scaled = orient_features_for_risk(self.scaler.transform(select_features(x, self.feature_indices)), self.feature_names)
        return _scores_to_labels(self._compute_scores(x_scaled), self.thresholds)

    def predict_scores(self, x):
        x_scaled = orient_features_for_risk(self.scaler.transform(select_features(x, self.feature_indices)), self.feature_names)
        return self._compute_scores(x_scaled)

    def _compute_scores(self, x_scaled: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("TOPSISClassifier must be fitted before prediction.")
        weighted = x_scaled * self.weights
        ideal_best = weighted.max(axis=0)
        ideal_worst = weighted.min(axis=0)
        dist_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
        dist_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
        return dist_worst / (dist_best + dist_worst + 1e-10)


class GRAClassifier(BaseEstimator, ClassifierMixin):
    """Grey relational analysis baseline."""

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        rho: float = 0.5,
        feature_indices: tuple[int, ...] | None = None,
    ):
        self.n_classes = n_classes
        self.rho = rho
        self.feature_indices = feature_indices
        self.weights: np.ndarray | None = None
        self.thresholds: list[float] | None = None
        self.scaler = MinMaxScaler()
        self.feature_names = feature_names_for_indices(feature_indices)

    def fit(self, x, y=None):
        del y
        x_selected = select_features(x, self.feature_indices)
        x_scaled = orient_features_for_risk(self.scaler.fit_transform(x_selected), self.feature_names)
        self.weights = prior_weights_for_indices(self.feature_indices)
        scores = self._compute_scores(x_scaled)
        self.thresholds = _percentile_thresholds(scores, self.n_classes)
        return self

    def predict(self, x):
        x_scaled = orient_features_for_risk(self.scaler.transform(select_features(x, self.feature_indices)), self.feature_names)
        return _scores_to_labels(self._compute_scores(x_scaled), self.thresholds)

    def predict_scores(self, x):
        x_scaled = orient_features_for_risk(self.scaler.transform(select_features(x, self.feature_indices)), self.feature_names)
        return self._compute_scores(x_scaled)

    def _compute_scores(self, x_scaled: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("GRAClassifier must be fitted before prediction.")
        reference = x_scaled.max(axis=0)
        diff = np.abs(x_scaled - reference)
        min_diff = diff.min()
        max_diff = diff.max()
        coefficients = (min_diff + self.rho * max_diff) / (diff + self.rho * max_diff + 1e-10)
        return (coefficients * self.weights).sum(axis=1)


class FuzzyComprehensiveEvaluator(BaseEstimator, ClassifierMixin):
    """Weighted fuzzy comprehensive evaluation baseline."""

    def __init__(self, n_classes: int = N_CLASSES, feature_indices: tuple[int, ...] | None = None):
        self.n_classes = n_classes
        self.feature_indices = feature_indices
        self.weights: np.ndarray | None = None
        self.thresholds: list[float] | None = None
        self.scaler = MinMaxScaler()
        self.feature_names = feature_names_for_indices(feature_indices)

    def fit(self, x, y=None):
        del y
        x_selected = select_features(x, self.feature_indices)
        x_scaled = orient_features_for_risk(self.scaler.fit_transform(x_selected), self.feature_names)
        self.weights = prior_weights_for_indices(self.feature_indices)
        scores = self._compute_scores(x_scaled)
        self.thresholds = _percentile_thresholds(scores, self.n_classes)
        return self

    def predict(self, x):
        x_scaled = orient_features_for_risk(self.scaler.transform(select_features(x, self.feature_indices)), self.feature_names)
        return _scores_to_labels(self._compute_scores(x_scaled), self.thresholds)

    def predict_scores(self, x):
        x_scaled = orient_features_for_risk(self.scaler.transform(select_features(x, self.feature_indices)), self.feature_names)
        return self._compute_scores(x_scaled)

    def _compute_scores(self, x_scaled: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("FuzzyComprehensiveEvaluator must be fitted before prediction.")
        return (x_scaled * self.weights).sum(axis=1)


class EntropyTOPSISClassifier(TOPSISClassifier):
    """TOPSIS with entropy-derived objective weights."""

    def __init__(self, n_classes: int = N_CLASSES, feature_indices: tuple[int, ...] | None = None):
        super().__init__(n_classes=n_classes, use_entropy_weight=True, feature_indices=feature_indices)


class CombinedWeightTOPSISClassifier(TOPSISClassifier):
    """TOPSIS with a blend of expert prior and entropy weights."""

    def __init__(self, n_classes: int = N_CLASSES, alpha: float = 0.5, feature_indices: tuple[int, ...] | None = None):
        super().__init__(n_classes=n_classes, use_entropy_weight=False, feature_indices=feature_indices)
        self.alpha = alpha

    def fit(self, x, y=None):
        del y
        x_selected = select_features(x, self.feature_indices)
        x_scaled = orient_features_for_risk(self.scaler.fit_transform(x_selected), self.feature_names)
        self.weights = combination_weight(x_scaled, prior_weights=prior_weights_for_indices(self.feature_indices), alpha=self.alpha)
        scores = self._compute_scores(x_scaled)
        self.thresholds = _percentile_thresholds(scores, self.n_classes)
        return self


class UrgencyEvaluator:
    """Transparent urgency baseline based on arrival time and defense pressure."""

    def __init__(self, n_urgency: int = N_URGENCY):
        self.n_urgency = n_urgency
        self.scaler = MinMaxScaler()
        self.thresholds: list[float] | None = None

    def fit(self, x, y=None):
        del y
        x_scaled = self.scaler.fit_transform(x)
        scores = self._compute_scores(x_scaled)
        self.thresholds = _percentile_thresholds(scores, self.n_urgency)
        return self

    def predict(self, x):
        x_scaled = self.scaler.transform(x)
        return _scores_to_labels(self._compute_scores(x_scaled), self.thresholds)

    @staticmethod
    def _compute_scores(x_scaled: np.ndarray) -> np.ndarray:
        distance = x_scaled[:, 8]
        velocity = x_scaled[:, 9]
        time_to_arrival = x_scaled[:, 11]
        defense = x_scaled[:, 13]
        asset_value = x_scaled[:, 14]
        track_confidence = x_scaled[:, 15]
        return (
            0.38 * (1.0 - time_to_arrival)
            + 0.14 * (1.0 - distance)
            + 0.12 * velocity
            + 0.16 * (1.0 - defense)
            + 0.12 * asset_value
            + 0.08 * (1.0 - track_confidence)
        )


class TraditionalDualTaskModel:
    """Wrap one threat scorer and one urgency scorer as a dual-task model."""

    def __init__(self, threat_model, urgency_model=None):
        self.threat_model = threat_model
        self.urgency_model = urgency_model if urgency_model is not None else UrgencyEvaluator()

    def fit(self, x, y_threat=None, y_urgency=None):
        self.threat_model.fit(x, y_threat)
        self.urgency_model.fit(x, y_urgency)
        return self

    def predict(self, x):
        return self.threat_model.predict(x), self.urgency_model.predict(x)


class TemporalHMMDualTaskModel:
    """Gaussian-emission HMM baseline for sequential threat assessment.

    The hidden states are ordinal threat or urgency labels. Emissions are
    diagonal Gaussian feature likelihoods and decoding uses Viterbi. This gives
    ATUAV a classical dynamic-assessment reference similar in spirit to
    the HMM baseline used in LSS target threat-assessment studies.
    """

    uses_sequence_input = True

    def __init__(self):
        self.threat_hmm = GaussianOrdinalHMM(n_states=N_CLASSES)
        self.urgency_hmm = GaussianOrdinalHMM(n_states=N_URGENCY)

    def fit(self, x, y_threat=None, y_urgency=None):
        sequences = _ensure_sequence_array(x)
        threat_seq = _ensure_label_sequence(y_threat, sequences.shape[1])
        urgency_seq = _ensure_label_sequence(y_urgency, sequences.shape[1])
        self.threat_hmm.fit(sequences, threat_seq)
        self.urgency_hmm.fit(sequences, urgency_seq)
        return self

    def fit_sequence(self, x, threat_seq, urgency_seq):
        return self.fit(x, threat_seq, urgency_seq)

    def predict(self, x):
        threat_seq, urgency_seq = self.predict_sequence(_ensure_sequence_array(x))
        return threat_seq[:, -1], urgency_seq[:, -1]

    def predict_sequence(self, x):
        sequences = _ensure_sequence_array(x)
        return self.threat_hmm.predict(sequences), self.urgency_hmm.predict(sequences)


class GaussianOrdinalHMM:
    """Minimal supervised HMM with diagonal Gaussian emissions."""

    def __init__(self, n_states: int, smoothing: float = 1.0, min_std: float = 0.04):
        self.n_states = n_states
        self.smoothing = float(smoothing)
        self.min_std = float(min_std)
        self.log_initial: np.ndarray | None = None
        self.log_transition: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    def fit(self, sequences: np.ndarray, labels: np.ndarray):
        sequences = _ensure_sequence_array(sequences)
        labels = _ensure_label_sequence(labels, sequences.shape[1])
        zero_labels = np.clip(labels.astype(np.int64) - 1, 0, self.n_states - 1)
        n_tracks, _, n_features = sequences.shape

        initial = np.full(self.n_states, self.smoothing, dtype=np.float64)
        transition = np.full((self.n_states, self.n_states), self.smoothing, dtype=np.float64)
        for track_idx in range(n_tracks):
            initial[zero_labels[track_idx, 0]] += 1.0
            for src, dst in zip(zero_labels[track_idx, :-1], zero_labels[track_idx, 1:]):
                transition[src, dst] += 1.0

        flat_x = sequences.reshape(-1, n_features)
        flat_y = zero_labels.reshape(-1)
        global_mean = flat_x.mean(axis=0)
        global_std = np.maximum(flat_x.std(axis=0), self.min_std)
        means = np.empty((self.n_states, n_features), dtype=np.float64)
        stds = np.empty((self.n_states, n_features), dtype=np.float64)
        for state in range(self.n_states):
            state_x = flat_x[flat_y == state]
            if len(state_x) == 0:
                means[state] = global_mean
                stds[state] = global_std
            else:
                means[state] = state_x.mean(axis=0)
                stds[state] = np.maximum(state_x.std(axis=0), self.min_std)

        self.log_initial = np.log(initial / initial.sum())
        self.log_transition = np.log(transition / transition.sum(axis=1, keepdims=True))
        self.means = means
        self.stds = stds
        return self

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        sequences = _ensure_sequence_array(sequences)
        if self.log_initial is None or self.log_transition is None or self.means is None or self.stds is None:
            raise RuntimeError("GaussianOrdinalHMM must be fitted before prediction.")
        decoded = [self._viterbi(track) + 1 for track in sequences]
        return np.asarray(decoded, dtype=np.int64)

    def _viterbi(self, track: np.ndarray) -> np.ndarray:
        emissions = self._log_emissions(track)
        n_steps = emissions.shape[0]
        scores = np.empty((n_steps, self.n_states), dtype=np.float64)
        backptr = np.zeros((n_steps, self.n_states), dtype=np.int64)
        scores[0] = self.log_initial + emissions[0]
        for step in range(1, n_steps):
            candidate = scores[step - 1, :, None] + self.log_transition
            backptr[step] = np.argmax(candidate, axis=0)
            scores[step] = np.max(candidate, axis=0) + emissions[step]

        states = np.zeros(n_steps, dtype=np.int64)
        states[-1] = int(np.argmax(scores[-1]))
        for step in range(n_steps - 2, -1, -1):
            states[step] = backptr[step + 1, states[step + 1]]
        return states

    def _log_emissions(self, track: np.ndarray) -> np.ndarray:
        diff = track[:, None, :] - self.means[None, :, :]
        var = np.square(self.stds[None, :, :])
        return -0.5 * np.sum(np.square(diff) / var + np.log(2.0 * np.pi * var), axis=2)


def _ensure_sequence_array(x) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 2:
        return values[:, None, :]
    if values.ndim != 3:
        raise ValueError(f"Expected 2D or 3D features for TemporalHMM, got {values.shape}")
    return values


def _ensure_label_sequence(labels, n_steps: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim == 1:
        return np.repeat(values[:, None], n_steps, axis=1)
    if values.ndim != 2:
        raise ValueError(f"Expected 1D or 2D labels for TemporalHMM, got {values.shape}")
    if values.shape[1] != n_steps:
        return values[:, :n_steps] if values.shape[1] > n_steps else np.repeat(values[:, -1:], n_steps, axis=1)
    return values


def get_traditional_models() -> dict[str, TraditionalDualTaskModel]:
    """Return all classical baselines used in assessment model groups."""
    return {
        TRADITIONAL_MODEL_NAMES[0]: TraditionalDualTaskModel(
            TOPSISClassifier(),
            TOPSISClassifier(n_classes=N_URGENCY, feature_indices=URGENCY_FEATURE_INDICES),
        ),
        TRADITIONAL_MODEL_NAMES[1]: TraditionalDualTaskModel(
            GRAClassifier(),
            GRAClassifier(n_classes=N_URGENCY, feature_indices=URGENCY_FEATURE_INDICES),
        ),
        TRADITIONAL_MODEL_NAMES[2]: TraditionalDualTaskModel(
            FuzzyComprehensiveEvaluator(),
            FuzzyComprehensiveEvaluator(n_classes=N_URGENCY, feature_indices=URGENCY_FEATURE_INDICES),
        ),
        TRADITIONAL_MODEL_NAMES[3]: TraditionalDualTaskModel(
            EntropyTOPSISClassifier(),
            EntropyTOPSISClassifier(n_classes=N_URGENCY, feature_indices=URGENCY_FEATURE_INDICES),
        ),
        TRADITIONAL_MODEL_NAMES[4]: TraditionalDualTaskModel(
            CombinedWeightTOPSISClassifier(),
            CombinedWeightTOPSISClassifier(n_classes=N_URGENCY, feature_indices=URGENCY_FEATURE_INDICES),
        ),
        TRADITIONAL_MODEL_NAMES[5]: TemporalHMMDualTaskModel(),
    }


def _percentile_thresholds(scores: np.ndarray, n_classes: int) -> list[float]:
    return [float(np.percentile(scores, 100.0 * idx / n_classes)) for idx in range(1, n_classes)]


def _scores_to_labels(scores: np.ndarray, thresholds: list[float] | None) -> np.ndarray:
    if thresholds is None:
        raise RuntimeError("The model must be fitted before prediction.")
    labels = np.ones(len(scores), dtype=np.int64)
    for threshold_idx, threshold in enumerate(thresholds):
        labels[scores >= threshold] = threshold_idx + 2
    return labels
