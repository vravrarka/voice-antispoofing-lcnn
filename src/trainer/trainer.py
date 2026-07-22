import torch
from tqdm.auto import tqdm
from src.metrics import EERAccumulator
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """
    def _evaluation_epoch(
        self,
        epoch,
        part,
        dataloader,
    ):
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        eer_accumulator = None
        if part == "eval":
            eer_accumulator = EERAccumulator()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
                if eer_accumulator is not None:
                    eer_accumulator.update(
                        logits=batch["logits"],
                        labels=batch["labels"],
                    )
                self.writer.set_step(
                    epoch * self.epoch_len,
                    part,
                )
        result = self.evaluation_metrics.result()
        if eer_accumulator is not None:
            eer_percent, eer_threshold = (
                eer_accumulator.compute()
            )
            result["eer"] = eer_percent
            self.logger.info(
                f"Eval EER threshold: "
                f"{eer_threshold:.6f}"
            )
        return result

    def _train_epoch(
        self,
        epoch,
    ):
        self._epoch_train_loss_sum = 0.0
        self._epoch_train_examples = 0
        logs = super()._train_epoch(epoch)
        if self._epoch_train_examples == 0:
            raise RuntimeError(
                "No training examples were processed."
            )
        train_loss = (
            self._epoch_train_loss_sum
            / self._epoch_train_examples
        )
        logs["loss"] = train_loss
        required_metrics = {
            "loss",
            "accuracy",
            "dev_loss",
            "dev_accuracy",
            "eval_loss",
            "eval_accuracy",
            "eval_eer",
        }
        missing_metrics = (
            required_metrics - set(logs)
        )
        if missing_metrics:
            raise KeyError(
                "Cannot log required WandB metrics. "
                f"Missing: {sorted(missing_metrics)}. "
                "Check that dataloaders are named "
                "'train', 'dev' and 'eval'."
            )
        if self.writer is not None:
            self.writer.add_epoch_metrics(
                epoch=epoch,
                metrics={
                    "train_loss": logs["loss"],
                    "train_accuracy": logs["accuracy"],
                    "dev_loss": logs["dev_loss"],
                    "dev_accuracy": logs["dev_accuracy"],
                    "eval_loss": logs["eval_loss"],
                    "eval_accuracy": logs["eval_accuracy"],
                    "eval_eer": logs["eval_eer"],
                },
            )
        return logs

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        batch_size = int(
            batch["labels"].shape[0]
        )
        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item(), n=batch_size,)
        if self.is_train:
            self._epoch_train_loss_sum += (
                batch["loss"].item() * batch_size
            )
            self._epoch_train_examples += batch_size

        for met in metric_funcs:
            metrics.update(met.name, met(**batch), n=batch_size,)
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
