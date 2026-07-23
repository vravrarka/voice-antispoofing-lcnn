import csv
from pathlib import Path
import torch
from tqdm.auto import tqdm
from src.metrics import bonafide_score_from_logits
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        checkpoint_path = config.inferencer.get(
            "from_pretrained"
        )
        assert skip_model_load or checkpoint_path is not None, (
            "Provide a checkpoint path or set "
            "skip_model_load=True."
        )
        self.config = config
        self.cfg_trainer = config.inferencer
        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms
        self.evaluation_dataloaders = dict(
            dataloaders
        )
        self.save_path = Path(save_path)
        self.metrics = metrics
        inference_metrics = list(
            self.metrics["inference"]
        )
        self.evaluation_metrics = MetricTracker(
            *[
                metric.name
                for metric in inference_metrics
            ],
            writer=None,
        )
        if not skip_model_load:
            self._from_pretrained(
                checkpoint_path
            )

    def run_inference(self) -> dict:
        part_logs = {}
        for part, dataloader in (
            self.evaluation_dataloaders.items()
        ):
            part_logs[part] = self._inference_part(
                part=part,
                dataloader=dataloader,
            )
        return part_logs

    def process_batch(
        self,
        batch,
        metrics,
    ) -> dict:
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        outputs = self.model(**batch)
        batch.update(outputs)
        for metric in self.metrics["inference"]:
            metrics.update(
                metric.name,
                metric(**batch),
                n=batch["labels"].shape[0],
            )
        return batch

    def _save_csv(
        self,
        part: str,
        prediction_rows: list[tuple[str, float]],
    ) -> Path:
        self.save_path.mkdir(
            parents=True,
            exist_ok=True,
        )
        csv_path = self.save_path / f"{part}.csv"
        utterance_ids = [
            utterance_id
            for utterance_id, _ in prediction_rows
        ]
        if len(utterance_ids) != len(
            set(utterance_ids)
        ):
            raise ValueError(
                "Duplicate utterance IDs found."
            )
        for _, score in prediction_rows:
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Invalid score: {score}"
                )
        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerows(prediction_rows)
        print(
            f"Saved {len(prediction_rows)} "
            f"predictions to {csv_path}"
        )
        return csv_path

    def _inference_part(
        self,
        part,
        dataloader,
    ) -> dict:
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        prediction_rows = []
        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch=batch,
                    metrics=self.evaluation_metrics,
                )
                scores_tensor = (
                    bonafide_score_from_logits(
                        batch["logits"]
                    )
                )
                if not bool(
                    torch.isfinite(
                        scores_tensor
                    ).all()
                ):
                    raise ValueError(
                        "Non-finite scores found."
                    )
                scores = (
                    scores_tensor
                    .detach()
                    .cpu()
                    .tolist()
                )
                utterance_ids = (
                    batch["utterance_id"]
                )
                if len(utterance_ids) != len(scores):
                    raise RuntimeError(
                        "Number of IDs and scores differs."
                    )
                prediction_rows.extend(
                    (
                        str(utterance_id),
                        float(score),
                    )
                    for utterance_id, score
                    in zip(
                        utterance_ids,
                        scores,
                    )
                )
        self._save_csv(
            part=part,
            prediction_rows=prediction_rows,
        )
        result = self.evaluation_metrics.result()
        result["num_predictions"] = len(
            prediction_rows
        )
        return result
