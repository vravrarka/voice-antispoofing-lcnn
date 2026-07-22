import torch
from torch import Tensor, nn

class AntiSpoofingCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.loss_function = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        **batch,
    ) -> dict[str, Tensor]:
        if logits.ndim != 2:
            raise ValueError(
                "Expected logits with shape [batch_size, 2], "
                f"but received {tuple(logits.shape)}."
            )
        if logits.shape[1] != 2:
            raise ValueError(
                "Expected exactly two output classes, "
                f"but received {logits.shape[1]}."
            )
        if labels.ndim != 1:
            raise ValueError(
                "Expected labels with shape [batch_size], "
                f"but received {tuple(labels.shape)}."
            )
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                "The number of logits and labels must match: "
                f"{logits.shape[0]} != {labels.shape[0]}."
            )
        labels = labels.to(
            device=logits.device,
            dtype=torch.long,
        )
        valid_labels = (labels == 0) | (labels == 1)
        if not bool(valid_labels.all()):
            raise ValueError(
                "Labels must contain only 0 (spoof) and 1 (bonafide)."
            )
        loss = self.loss_function(
            logits,
            labels,
        )
        return {
            "loss": loss,
        }
