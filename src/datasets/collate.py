import torch


def collate_fn(dataset_items: list[dict]) -> dict:
    
    result_batch = {}

    # example of collate_fn
    result_batch["features"] = torch.stack(
        [
            elem["features"] 
            for elem in dataset_items
        ],
        dim=0,
    )
    result_batch["labels"] = torch.tensor(
        [
            elem["labels"] 
            for elem in dataset_items
        ]
    )

    return result_batch
