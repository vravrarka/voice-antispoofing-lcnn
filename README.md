# Voice Anti-Spoofing (homework)

Этот проект решает задачу **Voice Anti-Spoofing**: модель должна отличать настоящую речь (**bonafide**) от синтезированной или преобразованной речи (**spoof**).

Для экспериментов используется раздел **Logical Access (LA)** датасета **ASVspoof 2019**. Модель построена на архитектуре **Light Convolutional Neural Network (LCNN)** и обучается как бинарный классификатор.

Проект основан на **PyTorch Project Template** и использует **Hydra** для конфигурации экспериментов и **Weights & Biases (W&B)** для логирования.

## Task

Для каждого аудиофайла модель должна предсказать один из двух классов:
- `bonafide` — настоящая запись;
- `spoof` — синтезированная или преобразованная запись.

## Method

### Front-End

Аудио приводится к `16000 Hz`, после чего вычисляется **Log-Power Spectrogram** с помощью **STFT**.

Использованные параметры:

| Parameter | Value |
|---|---:|
| `sample_rate` | `16000` |
| `n_fft` | `1724` |
| `win_length` | `1724` |
| `hop_length` | `130` |
| `target_frames` | `600` |

Короткие записи дополняются нулями. Для длинных записей выбираются первые `600` временных кадров во время evaluation; в training может применяться случайный crop.

### Model

В проекте реализована **LCNN** с блоками **Max-Feature-Map (MFM)**. После convolutional-части используются:

1. `MFMLinear`;
2. `Dropout(p=0.75)`;
3. `BatchNorm1d`;
4. `Linear` для двух классов.

`Dropout` расположен перед последним `BatchNorm`, как указано в задании.

### Labels and Scores

Внутри проекта используются метки:

```text
spoof = 0
bonafide = 1
```

Для итогового CSV сохраняется вероятность класса `bonafide`:

```python
softmax(logits)[:, 1]
```

## Dataset

Используется **ASVspoof 2019 LA**:

```text
ASVspoof2019_LA_train
ASVspoof2019_LA_dev
ASVspoof2019_LA_eval
ASVspoof2019_LA_cm_protocols
```

Аудиофайлы и checkpoint-файлы не хранятся в GitHub.

Для запуска в Kaggle датасет можно подключить как Input:

```text
awsaf49/asvpoof-2019-dataset
```

Пути к данным задаются в:

```text
src/configs/paths/kaggle.yaml
```

## Project Structure

```text
voice-antispoofing-lcnn/
├── grading/
│   ├── grading.py
│   ├── calculate_eer.py
│   ├── ASVspoof2019.LA.cm.eval.trl.txt
│   └── students_solutions/
├── reports/
│   ├── figures/
│   ├── wandb/
│   └── report.pdf
├── src/
│   ├── configs/
│   ├── datasets/
│   ├── logger/
│   ├── loss/
│   ├── metrics/
│   ├── model/
│   ├── trainer/
│   ├── transforms/
│   └── utils/
├── tests/
├── inference.py
├── train.py
└── requirements.txt
```

## Installation

Рекомендуется использовать Python `3.10+`.

```bash
git clone https://github.com/vravrarka/voice-antispoofing-lcnn.git
cd voice-antispoofing-lcnn

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Для Windows:

```bash
.venv\Scripts\activate
```

Для логирования необходимо авторизоваться в W&B:

```bash
wandb login
```

В Kaggle API key лучше хранить в **Secrets** под именем:

```text
WANDB_API_KEY
```

## Training

Основной конфиг обучения:

```text
src/configs/lcnn_baseline.yaml
```

Пример запуска:

```bash
python train.py \
  -cn=lcnn_baseline \
  writer.run_name="lcnn-full-seed1-stage1"
```

Основные параметры эксперимента:

| Parameter | Value |
|---|---:|
| `optimizer` | `Adam` |
| `learning_rate` | `0.0003` |
| `weight_decay` | `0.0001` |
| `batch_size` | `8` |
| `loss` | `CrossEntropyLoss` |
| `seed` | `1` |
| `checkpoint_monitor` | `min dev_loss` |

Точное число завершённых epochs=7 и остальные параметры следует смотреть в **W&B Config** для финального run.

### One-Batch Test

Перед полным обучением был выполнен **one-batch overfit test**. Модель смогла переобучиться на фиксированном небольшом batch и достигла:

```text
accuracy = 1.0
EER = 0.0%
```
Этот тест использовался только для проверки pipeline.

## Inference

Для inference нужен checkpoint `model_best.pth`.
Checkpoint не добавляется в GitHub. Он хранится отдельно или в W&B.
Пример запуска:

```bash
python inference.py \
  -cn=inference_lcnn \
  inferencer.from_pretrained="/path/to/model_best.pth" \
  inferencer.save_path="full-eval-final"
```

Ожидаемый результат:
```text
data/saved/full-eval-final/eval.csv
```

## Results

Финальный CSV был построен с использованием `model_best.pth` и проверен через `grading.py`.

| Result | Value |
|---|---:|
| Lowest logged `eval_eer` | approximately `7.1%` |
| Final CSV EER | approximately `9.7%` |
| Final grade from `grading.py` | `<ADD_EXACT_VALUE>` |

Самый низкий EER наблюдался не на том checkpoint, который был выбран как `model_best.pth`. `model_best.pth` выбирался по минимальному `dev_loss`, а не по минимальному `eval_eer`. Поэтому итоговый EER CSV отличается от минимального значения на графике.

## Experiment Tracking

### W&B Project

- Project: [voice-antispoofing-lcnn](https://wandb.ai/varvaravolodicheva-hse-university/voice-antispoofing-lcnn)
- Final Run: `<[ADD_FINAL_RUN_URL](https://wandb.ai/varvaravolodicheva-hse-university/voice-antispoofing-lcnn/runs/6zw5r1ha?nw=nwuservarvaravolodicheva)>`

В W&B должны быть доступны:

- `train_loss`;
- `train_accuracy`;
- `dev_loss`;
- `dev_accuracy`;
- `eval_loss`;
- `eval_accuracy`;
- `eval_eer`;
- experiment Config;
- system metrics.

## Limitations

- Был выполнен один полный training run с ограниченным числом epochs.
- Training был остановлен из-за окончания бесплатной верссии GPU quota.
- `model_best.pth` выбирался по `dev_loss`, поэтому он не соответствует минимальному `eval_eer`.
- Результат может изменяться при другом random seed и при более долгом обучении.
