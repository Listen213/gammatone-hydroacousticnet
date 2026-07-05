````bash
```markdown
# Gammatone-HydroacousticNet: Underwater Acoustic Classification with Limited Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository provides the official implementation of **Gammatone-HydroacousticNet**, an auditory-inspired feature fusion approach for underwater acoustic classification with limited data.

---

## 📌 Overview

Underwater acoustic classification faces two major challenges: complex marine environments and scarce annotated data. Our proposed framework:

- **Gammatone Filter Bank**: 128-channel auditory filter mimicking human cochlea (20–8000 Hz), extracting robust time-frequency features.
- **HydroacousticNet**: Modernized ConvNeXt-based backbone with layer-wise learning rate decay (LLRD) and tailored augmentation.
- **Performance**: Achieves **93.6%** accuracy on ShipsEar with only **7%** training data, and **96.6%** on DeepShip.

---

## 🗂️ Repository Structure

```
.
├── full_pipeline.py          # Full pipeline: audio → Gammatone images → dataset split → small-sample list
├── main.py                   # Training script (ConvNeXt)
├── test.py                   # Testing script with detailed metrics
├── models/
│   ├── convnext.py           # ConvNeXt model definition
│   └── convnext_isotropic.py # Isotropic ConvNeXt variant
├── datasets.py               # Dataset loading and transforms
├── engine.py                 # Training/evaluation engine
├── optim_factory.py          # Optimizer and layer decay
├── utils.py                  # Utilities (logging, metrics, checkpointing)
├── run_with_submitit.py      # SLURM cluster submission
├── analyze_dataset.py        # Dataset distribution analysis
├── requirements.txt          # Python dependencies
├── test_results.json         # Example test results
├── train.log                 # Example training log
└── README.md                 # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Listen213/gammatone-hydroacousticnet.git
cd gammatone-hydroacousticnet

# Create a virtual environment (optional but recommended)
conda create -n hydroacoustic python=3.9
conda activate hydroacoustic

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your audio files and metadata CSV in the following structure:

```
ShipsEar/
├── audio1.wav
├── audio2.wav
└── ...
shipsEarMeta_90.csv      # Contains 'filename' and 'type' columns
```

The CSV format should include:
- `filename`: audio file name
- `type`: vessel type (e.g., "Motorboat", "Fishboat", etc.)

### 3. Generate Gammatone Features and Split Dataset

Run the full pipeline to:
- Convert audio to Gammatone time-frequency images (224×224 PNG)
- Split into train/val/test (7:1.5:1.5)
- Extract small-sample filenames (7% of training)

```bash
python full_pipeline.py
```

**Outputs:**
- `audio_class_gammatone_images/` – raw feature maps
- `ships_gammatone_dataset_split/` – train/val/test folders
- `small_sample_file_list.txt` – filenames of 7% training samples

### 4. Train the Model

```bash
python main.py \
    --data_set image_folder \
    --data_path ships_gammatone_dataset_split/train \
    --eval_data_path ships_gammatone_dataset_split/val \
    --nb_classes 5 \
    --input_size 224 \
    --batch_size 32 \
    --epochs 200 \
    --model convnext_tiny \
    --lr 1e-3 \
    --output_dir ./outputs \
    --imagenet_default_mean_and_std False
```

**Key Arguments:**
| Argument | Description |
|----------|-------------|
| `--data_set` | `image_folder` for custom datasets |
| `--data_path` | Training set path |
| `--eval_data_path` | Validation set path |
| `--nb_classes` | Number of classes (5 for ShipsEar) |
| `--model` | `convnext_tiny`, `convnext_small`, etc. |
| `--lr` | Initial learning rate |
| `--output_dir` | Checkpoint saving directory |

### 5. Evaluate the Model

```bash
# Test on the final test set
python test.py
```

The script outputs:
- Overall accuracy
- Per-class accuracy
- Confusion matrix (error analysis)
- JSON results saved to `test_results.json`

**Optional:** Test a single image:
```bash
python test.py /path/to/your/image.png
```

---

## 📊 Results

### ShipsEar Dataset (7% Training Data)

| Model | Accuracy | F1-score |
|-------|----------|----------|
| Mel + ResNet-18 | 86.7% | 86.7% |
| Gammatone + ResNet-18 | 87.7% | 87.5% |
| Mel + HydroacousticNet | 91.9% | 91.9% |
| **Gammatone + HydroacousticNet (Ours)** | **93.6%** | **93.4%** |

### DeepShip Subset (7% Training Data)

| Feature | Accuracy | F1-score |
|---------|----------|----------|
| Mel | 92.5% | 92.4% |
| CQT | 90.8% | 90.6% |
| STFT | 89.0% | 88.9% |
| **Gammatone (Ours)** | **96.6%** | **96.5%** |

### Per-Class Performance (ShipsEar)

| Class | Accuracy |
|-------|----------|
| A: Large Merchant Ship | 97.2% |
| B: Small Speedboat | 88.6% |
| C: Fishing Boat | 87.9% |
| D: Ferry | 92.4% |
| E: Ambient Noise | 99.5% |

---

## 🔧 Customization

### Using Your Own Dataset

1. **Prepare data**: Organize images in a folder structure with class subfolders.
2. **Update paths** in `full_pipeline.py` and `main.py`.
3. **Adjust `nb_classes`** in training command.

### Adjusting Layer-wise Learning Rate Decay

In `main.py`, modify:
```python
--layer_decay 0.9  # Decay factor (0.9 = exponential decay)
```

### Data Augmentation

Modify `datasets.py` → `build_transform()`:
```python
transforms.RandomResizedCrop(args.input_size, scale=(0.85, 1.0)),
transforms.RandomHorizontalFlip(p=0.3),
transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
```

---

## 📝 Code Availability Statement

All code is released under the **MIT License**. The model checkpoints and datasets are publicly available:

- **ShipsEar Dataset**: [https://github.com/DavidSanAndres/ShipsEar](https://github.com/DavidSanAndres/ShipsEar)
- **DeepShip Dataset**: [https://github.com/irfankundi/DeepShip](https://github.com/irfankundi/DeepShip)
- **Pre-trained ConvNeXt weights**: Loaded automatically via `timm` or Meta's official URLs.

---

## 🧪 Reproducibility

To reproduce our results:

1. **Fix random seed**: Set `RANDOM_SEED = 42` in `full_pipeline.py`.
2. **Use provided splits**: The dataset split is deterministic.
3. **Training**: Use the exact commands in the Quick Start section.
4. **Evaluation**: Run `test.py` on the final test set.


## 🤝 Contributing

Issues and pull requests are welcome! Please open an issue for discussion before submitting major changes.

---

## 📧 Contact

- **Qianxing Wei** – 15736521015@163.com
- **Chaochao Sun (Corresponding)** – pop-scc@163.com

---

## 🙏 Acknowledgements

This work builds upon:
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) by Meta Research
- [ShipsEar](https://github.com/DavidSanAndres/ShipsEar) dataset
- [librosa](https://librosa.org/) audio processing library

---

## 📜 License

MIT License – see [LICENSE](LICENSE) file for details.
```
````