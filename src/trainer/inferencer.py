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
        assert (
            skip_model_load
            or config.inferencer.get(
                "from_pretrained"
            )
            is not None
        ), (
            "Provide checkpoint path or set "
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
        self.save_path = (
            Path(save_path)
            if save_path is not None
            else None
        )
        self.metrics = metrics
        inference_metrics = []
        if (
            self.metrics is not None
            and self.metrics.get("inference")
            is not None
        ):
            inference_metrics = list(
                self.metrics["inference"]
            )
        if inference_metrics:
            self.evaluation_metrics = (
                MetricTracker(
                    *[
                        metric.name
                        for metric
                        in inference_metrics
                    ],
                    writer=None,
                )
            )
        else:
            self.evaluation_metrics = None
        if not skip_model_load:
            self._from_pretrained(
                config.inferencer.from_pretrained
            )

    def run_inference(self) -> dict:
        """Run inference for every configured partition."""
        part_logs = {}
        for part, dataloader in (
            self.evaluation_dataloaders.items()
        ):
            part_logs[part] = (
                self._inference_part(
                    part=part,
                    dataloader=dataloader,
                )
            )
        return part_logs

    def process_batch(
        self,
        batch_idx,
        batch,
        metrics,
        part,
    ):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        outputs = self.model(**batch)
        batch.update(outputs)
        if metrics is not None:
            for metric in self.metrics["inference"]:
                metrics.update(
                    metric.name,
                    metric(**batch),
                )
        return batch

    def _save_predictions(
        self,
        part: str,
        prediction_rows: list[tuple[str, float]],
    ) -> Path | None:
        """Save utterance IDs and bona fide scores."""
        if self.save_path is None:
            return None
        self.save_path.mkdir(
            parents=True,
            exist_ok=True,
        )
        csv_path = self.save_path / f"{part}.csv"
        utterance_ids = [
            utterance_id
            for utterance_id, _
            in prediction_rows
        ]
        if len(utterance_ids) != len(
            set(utterance_ids)
        ):
            raise ValueError(
                "Duplicate utterance IDs were found "
                "during inference."
            )
        for _, score in prediction_rows:
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    "Prediction score must be "
                    f"between 0 and 1, got {score}."
                )
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
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
        """Run inference for one data partition."""
        self.is_train = False
        self.model.eval()
        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()
        prediction_rows = []
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    metrics=(
                        self.evaluation_metrics
                    ),
                    part=part,
                )
                scores = (
                    bonafide_score_from_logits(
                        batch["logits"]
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                utterance_ids = (
                    batch["utterance_id"]
                )
                if len(utterance_ids) != len(scores):
                    raise RuntimeError(
                        "Number of utterance IDs "
                        "does not match the number "
                        "of prediction scores."
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
        self._save_predictions(
            part=part,
            prediction_rows=prediction_rows,
        )
        if self.evaluation_metrics is None:
            return {
                "num_predictions": len(
                    prediction_rows
                )
            }
        return self.evaluation_metrics.result()
