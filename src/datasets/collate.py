import torch

def collate_fn(dataset_items: list[dict]) -> dict:
    return {
        "features": torch.stack(
            [
                item["features"]
                for item in dataset_items
            ],
            dim=0,
        ),
        "labels": torch.tensor(
            [
                item["labels"]
                for item in dataset_items
            ],
            dtype=torch.long,
        ),
        "utterance_id": [
            item["utterance_id"]
            for item in dataset_items
        ],
    }
