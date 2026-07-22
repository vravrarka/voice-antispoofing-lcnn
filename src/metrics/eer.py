import torch
from torch import Tensor
from src.metrics.calculate_eer import compute_eer

SPOOF_LABEL = 0
BONAFIDE_LABEL = 1

def bonafide_score_from_logits(
    logits: Tensor,
) -> Tensor:
    return torch.softmax(
        logits,
        dim=1,
    )[:, BONAFIDE_LABEL]

def calculate_eer_metric(
    scores: Tensor,
    labels: Tensor,
) -> tuple[float, float]:
    scores = (
        scores.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    labels = (
        labels.detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    bonafide_scores = scores[
        labels == BONAFIDE_LABEL
    ]
    spoof_scores = scores[
        labels == SPOOF_LABEL
    ]
    if bonafide_scores.size == 0:
        raise ValueError(
            "No bonafide samples were provided."
        )
    if spoof_scores.size == 0:
        raise ValueError(
            "No spoof samples were provided."
        )
    eer, threshold = compute_eer(
        bonafide_scores,
        spoof_scores,
    )
    return float(eer * 100), float(threshold)

class EERAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.scores = []
        self.labels = []

    def update(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> None:
        scores = bonafide_score_from_logits(
            logits
        )
        self.scores.append(
            scores.detach().cpu()
        )
        self.labels.append(
            labels.detach().cpu()
        )

    def compute(self) -> tuple[float, float]:
        if not self.scores:
            raise RuntimeError(
                "No evaluation batches were added."
            )
        scores = torch.cat(
            self.scores,
            dim=0,
        )
        labels = torch.cat(
            self.labels,
            dim=0,
        )
        return calculate_eer_metric(
            scores=scores,
            labels=labels,
        )
