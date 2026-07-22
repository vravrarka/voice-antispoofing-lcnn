from pathlib import Path

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


SPOOF_LABEL = 0
BONAFIDE_LABEL = 1

class ASVspoofDataset(Dataset):
    def __init__(
        self,
        audio_dir: str,
        protocol_path: str,
        feature_extractor,
        training: bool = False,
        sample_rate: int = 16000,
        limit: int | None = None,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.protocol_path = Path(protocol_path)
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.training = training
        if not self.audio_dir.exists():
            raise FileNotFoundError(
                f"Audio directory does not exist: "
                f"{self.audio_dir}"
            )
        if not self.protocol_path.exists():
            raise FileNotFoundError(
                f"Protocol file does not exist: "
                f"{self.protocol_path}"
            )
        if hasattr(self.feature_extractor, "train"):
            self.feature_extractor.train(training)
        self.records = self._read_protocol()
        if limit is not None:
            self.records = self.records[:limit]
        if not self.records:
            raise RuntimeError(
                f"No records were read from "
                f"{self.protocol_path}"
            )
    def _read_protocol(self) -> list[dict]:
        records = []
        with self.protocol_path.open(
            "r",
            encoding="utf-8",
        ) as protocol_file:
            for line_number, line in enumerate(
                protocol_file,
                start=1,
            ):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 2:
                    raise ValueError(
                        "Invalid protocol line "
                        f"{line_number}: {line!r}"
                    )
                utterance_id = parts[1]
                label_name = parts[-1].lower()
                if label_name == "bonafide":
                    label = BONAFIDE_LABEL
                elif label_name == "spoof":
                    label = SPOOF_LABEL
                else:
                    raise ValueError(
                        "Unknown label "
                        f"{label_name!r} on protocol "
                        f"line {line_number}."
                    )
                audio_path = (
                    self.audio_dir
                    / f"{utterance_id}.flac"
                )
                records.append(
                    {
                        "utterance_id": utterance_id,
                        "audio_path": audio_path,
                        "label": label,
                    }
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_waveform(
        self,
        audio_path: Path,
    ) -> Tensor:
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file does not exist: "
                f"{audio_path}"
            )
        waveform, source_sample_rate = (
            torchaudio.load(audio_path)
        )
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )
        if source_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=source_sample_rate,
                new_freq=self.sample_rate,
            )
        return waveform

    def __getitem__(
        self,
        index: int,
    ) -> dict:
        record = self.records[index]
        waveform = self._load_waveform(
            record["audio_path"]
        )
        features = self.feature_extractor(
            waveform
        )
        if features.ndim != 3:
            raise ValueError(
                "Feature extractor must return "
                "[channels, frequencies, frames], "
                f"received {tuple(features.shape)}."
            )
        return {
            "features": features.float(),
            "labels": int(record["label"]),
            "utterance_id": record["utterance_id"],
        }
    
