import torch
from src.model.lcnn import LCNN, max_feature_map

def test_max_feature_map() -> None:
    features = torch.tensor(
        [
            [
                1.0,
                5.0,
                3.0,
                2.0,
            ]
        ]
    )
    result = max_feature_map(features)
    expected = torch.tensor(
        [
            [
                3.0,
                5.0,
            ]
        ]
    )
    assert torch.equal(result, expected)

def test_convolutional_part_reduces_spatial_size() -> None:
    model = LCNN()
    model.eval()
    features = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        result = model.feature_extractor(features)
    assert result.shape == (1, 32, 4, 4)

def test_parameter_count() -> None:
    model = LCNN()
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    assert parameter_count == 10_198_818
