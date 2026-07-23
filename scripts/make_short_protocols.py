import random
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(
    "/kaggle/input/datasets/awsaf49/"
    "asvpoof-2019-dataset/LA/LA"
)

PROTOCOL_DIR = (
    DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
)

OUTPUT_DIR = Path("data/debug")

def create_balanced_protocol(
    source_path: Path,
    output_path: Path,
    bonafide_count: int,
    spoof_count: int,
    seed: int,
) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Protocol was not found: {source_path}"
        )
    lines = [
        line.strip()
        for line in source_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    bonafide_lines = [
        line
        for line in lines
        if line.split()[-1].lower() == "bonafide"
    ]
    spoof_lines = [
        line
        for line in lines
        if line.split()[-1].lower() == "spoof"
    ]
    if len(bonafide_lines) < bonafide_count:
        raise RuntimeError(
            f"Not enough bonafide samples in {source_path}"
        )
    if len(spoof_lines) < spoof_count:
        raise RuntimeError(
            f"Not enough spoof samples in {source_path}"
        )
    random_generator = random.Random(seed)
    selected_lines = (
        random_generator.sample(
            bonafide_lines,
            bonafide_count,
        )
        + random_generator.sample(
            spoof_lines,
            spoof_count,
        )
    )
    random_generator.shuffle(selected_lines)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        "\n".join(selected_lines) + "\n",
        encoding="utf-8",
    )
    labels = Counter(
        line.split()[-1].lower()
        for line in selected_lines
    )
    print(
        f"{output_path}: "
        f"{len(selected_lines)} samples, {labels}"
    )

def main() -> None:
    create_balanced_protocol(
        source_path=(
            PROTOCOL_DIR
            / "ASVspoof2019.LA.cm.train.trn.txt"
        ),
        output_path=(
            OUTPUT_DIR / "short_train_protocol.txt"
        ),
        bonafide_count=256,
        spoof_count=256,
        seed=42,
    )
    create_balanced_protocol(
        source_path=(
            PROTOCOL_DIR
            / "ASVspoof2019.LA.cm.dev.trl.txt"
        ),
        output_path=(
            OUTPUT_DIR / "short_dev_protocol.txt"
        ),
        bonafide_count=128,
        spoof_count=128,
        seed=43,
    )
    create_balanced_protocol(
        source_path=(
            PROTOCOL_DIR
            / "ASVspoof2019.LA.cm.eval.trl.txt"
        ),
        output_path=(
            OUTPUT_DIR / "short_eval_protocol.txt"
        ),
        bonafide_count=128,
        spoof_count=128,
        seed=44,
    )

if __name__ == "__main__":
    main()
