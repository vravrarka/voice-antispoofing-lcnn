from collections import OrderedDict

import torch
from torch import Tensor, nn

def max_feature_map(x: Tensor) -> Tensor:
    if x.ndim < 2:
        raise ValueError(
            "MFM expects a tensor with at least two dimensions."
        )
    if x.shape[1] % 2 != 0:
        raise ValueError(
            "The feature dimension passed to MFM must be even, "
            f"but received {x.shape[1]}."
        )
    first_half, second_half = x.chunk(2, dim=1)
    return torch.maximum(first_half, second_half)

class MFMConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return max_feature_map(self.conv(x))

class MFMLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=2 * out_features,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return max_feature_map(self.linear(x))


class LCNN(nn.Module):
    input_shape = (1, 863, 600)
    feature_map_shape = (32, 53, 37)
    flattened_features = 32 * 53 * 37

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.75,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError(
                "This LCNN is configured for binary anti-spoofing "
                f"classification, but num_classes={num_classes}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), received {dropout}."
            )
        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "mfm_conv1",
                        MFMConv2d(
                            in_channels=1,
                            out_channels=32,
                            kernel_size=5,
                            stride=1,
                            padding=2,
                        ),
                    ),
                    (
                        "pool3",
                        nn.MaxPool2d(
                            kernel_size=2,
                            stride=2,
                        ),
                    ),
                    (
                        "mfm_conv4",
                        MFMConv2d(
                            in_channels=32,
                            out_channels=32,
                            kernel_size=1,
                        ),
                    ),
                    ("batch_norm6", nn.BatchNorm2d(32)),
                    (
                        "mfm_conv7",
                        MFMConv2d(
                            in_channels=32,
                            out_channels=48,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                    (
                        "pool9",
                        nn.MaxPool2d(
                            kernel_size=2,
                            stride=2,
                        ),
                    ),
                    ("batch_norm10", nn.BatchNorm2d(48)),
                    (
                        "mfm_conv11",
                        MFMConv2d(
                            in_channels=48,
                            out_channels=48,
                            kernel_size=1,
                        ),
                    ),
                    ("batch_norm13", nn.BatchNorm2d(48)),
                    (
                        "mfm_conv14",
                        MFMConv2d(
                            in_channels=48,
                            out_channels=64,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                    (
                        "pool16",
                        nn.MaxPool2d(
                            kernel_size=2,
                            stride=2,
                        ),
                    ),
                    (
                        "mfm_conv17",
                        MFMConv2d(
                            in_channels=64,
                            out_channels=64,
                            kernel_size=1,
                        ),
                    ),
                    ("batch_norm19", nn.BatchNorm2d(64)),
                    (
                        "mfm_conv20",
                        MFMConv2d(
                            in_channels=64,
                            out_channels=32,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                    ("batch_norm22", nn.BatchNorm2d(32)),
                    (
                        "mfm_conv23",
                        MFMConv2d(
                            in_channels=32,
                            out_channels=32,
                            kernel_size=1,
                        ),
                    ),
                    ("batch_norm25", nn.BatchNorm2d(32)),
                    (
                        "mfm_conv26",
                        MFMConv2d(
                            in_channels=32,
                            out_channels=32,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                    (
                        "pool28",
                        nn.MaxPool2d(
                            kernel_size=2,
                            stride=2,
                        ),
                    ),
                ]
            )
        )

        self.fc_mfm = MFMLinear(
            in_features=self.flattened_features,
            out_features=80,
        )

        self.dropout = nn.Dropout(p=dropout)

        self.final_batch_norm = nn.BatchNorm1d(80)

        self.classifier = nn.Linear(
            in_features=80,
            out_features=num_classes,
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def extract_feature_map(self, features: Tensor) -> Tensor:
        return self.feature_extractor(features)

    def forward(
        self,
        features: Tensor,
        **batch,
    ) -> dict[str, Tensor]:
        if features.ndim != 4:
            raise ValueError(
                "LCNN expects a four-dimensional tensor "
                "[batch, channel, frequency, time], "
                f"but received shape {tuple(features.shape)}."
            )
        actual_shape = tuple(features.shape[1:])
        if actual_shape != self.input_shape:
            raise ValueError(
                f"LCNN expects input shape [B, {self.input_shape[0]}, "
                f"{self.input_shape[1]}, {self.input_shape[2]}], "
                f"but received {tuple(features.shape)}."
            )
        feature_map = self.extract_feature_map(features)
        expected_feature_map_shape = self.feature_map_shape
        actual_feature_map_shape = tuple(feature_map.shape[1:])
        if actual_feature_map_shape != expected_feature_map_shape:
            raise RuntimeError(
                "Unexpected convolutional output shape. "
                f"Expected [B, {expected_feature_map_shape}], "
                f"received {tuple(feature_map.shape)}."
            )
        flattened = torch.flatten(
            feature_map,
            start_dim=1,
        )
        embedding = self.fc_mfm(flattened)
        embedding = self.dropout(embedding)
        embedding = self.final_batch_norm(embedding)
        logits = self.classifier(embedding)
        return {
            "logits": logits,
            "embeddings": embedding,
        }
