from dataclasses import dataclass
from src.metrics.calculate_eer import (
    compute_eer as official_compute_eer,
)

BONAFIDE_LABEL = 1
SPOOF_LABEL = 0

@dataclass(frozen=True)
class EERResult:
    eer_percent: float
    threshold: float

def bonafide_score_from_logits(
    logits: Tensor,
) -> Tensor:
    if logits.ndim != 2:
        raise ValueError(
            "Expected logits with shape [batch_size, 2], "
            f"but received {tuple(logits.shape)}."
        )
    if logits.shape[1] != 2:
        raise ValueError(
            "Expected two logits per object, "
            f"but received {logits.shape[1]}."
        )
    probabilities = torch.softmax(
        logits,
        dim=1,
        dtype=torch.float32,
    )
    return probabilities[:, BONAFIDE_LABEL]

def _to_numpy_1d(
    values: Tensor | np.ndarray,
    name: str,
) -> np.ndarray:
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    else:
        values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, "
            f"but received shape {values.shape}."
        )
    return values

def calculate_eer_metric(
    scores: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
) -> EERResult:
    scores_array = _to_numpy_1d(
        scores,
        name="scores",
    ).astype(np.float64, copy=False)
    labels_array = _to_numpy_1d(
        labels,
        name="labels",
    ).astype(np.int64, copy=False)
    if scores_array.shape[0] != labels_array.shape[0]:
        raise ValueError(
            "The number of scores and labels must match: "
            f"{scores_array.shape[0]} != {labels_array.shape[0]}."
        )
    if not np.isfinite(scores_array).all():
        raise ValueError(
            "Scores contain NaN or Inf."
        )
    valid_labels = np.isin(
        labels_array,
        [SPOOF_LABEL, BONAFIDE_LABEL],
    )
    if not valid_labels.all():
        raise ValueError(
            "Labels must contain only 0 (spoof) and 1 (bonafide)."
        )
    bonafide_scores = scores_array[
        labels_array == BONAFIDE_LABEL
    ]
    spoof_scores = scores_array[
        labels_array == SPOOF_LABEL
    ]
    if bonafide_scores.size == 0:
        raise ValueError(
            "Cannot calculate EER without bonafide examples."
        )
    if spoof_scores.size == 0:
        raise ValueError(
            "Cannot calculate EER without spoof examples."
        )
    eer_fraction, threshold = official_compute_eer(
        bonafide_scores,
        spoof_scores,
    )
    return EERResult(
        eer_percent=float(eer_fraction * 100.0),
        threshold=float(threshold),
    )

def calculate_eer_from_logits(
    logits: Tensor,
    labels: Tensor,
) -> EERResult:
    scores = bonafide_score_from_logits(logits)
    return calculate_eer_metric(
        scores=scores,
        labels=labels,
    )

class EERAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._scores: list[Tensor] = []
        self._labels: list[Tensor] = []

    def update(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> None:
        scores = bonafide_score_from_logits(
            logits
        )
        self._scores.append(
            scores.detach().cpu()
        )
        self._labels.append(
            labels.detach().cpu().long()
        )

    def compute(self) -> EERResult:
        if not self._scores:
            raise RuntimeError(
                "No batches were added to the EER accumulator."
            )
        scores = torch.cat(
            self._scores,
            dim=0,
        )
        labels = torch.cat(
            self._labels,
            dim=0,
        )
        return calculate_eer_metric(
            scores=scores,
            labels=labels,
        )
