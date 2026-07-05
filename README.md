# Gammatone-HydroacousticNet: Underwater Acoustic Classification with Limited Data
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the official open-source implementation of **Gammatone-HydroacousticNet**, an auditory-inspired feature fusion framework designed for small-sample underwater acoustic vessel classification, which is submitted to *Applied Intelligence*.

## 📌 Overview
Underwater acoustic target classification suffers from two core practical bottlenecks: severe marine background noise and extreme scarcity of annotated underwater audio data. Traditional Mel-spectrum based methods lose critical low-frequency spectral details of vessel noise, while standard CNNs easily overfit when trained with only a tiny proportion of labeled samples. To solve this problem, we propose a synergistic two-stage framework:
1. **128-channel Gammatone Filter Bank**: Simulates human cochlear ERB frequency characteristics, generates high-resolution low-frequency time-frequency feature maps with strong anti-noise ability, covering the effective frequency band 20–8000 Hz of ship radiated noise.
2. **HydroacousticNet**: A modified ConvNeXt backbone adapted for single-channel auditory spectrogram input, integrated with exponential layer-wise learning rate decay (LLRD) and underwater acoustic tailored data augmentation to mitigate small-sample overfitting.

### Core Experimental Performance
- ShipsEar dataset (only 7% training samples): 93.6% overall classification accuracy
- Cross-domain DeepShip subset: 96.6% accuracy, demonstrating robust generalization across different recording environments and vessel types

### Reproducibility Statement
All scripts strictly fix random seed = 42 to eliminate random fluctuation. The 7% train / 1.5% val fixed split scheme in the paper is fully implemented in `full_pipeline.py`. All source codes, feature extraction pipelines, training & evaluation codes are fully open for peer review.

---

## 🗂️ Real Repository File Structure
```
.
├── LICENSE                   # MIT open-source license file
├── requirements.txt          # All python environment dependencies
├── README.md                 # This document
├── full_pipeline.py          # End-to-end audio preprocessing: WAV → Gammatone spectrogram + fixed dataset split (7% small sample)
├── analyze_dataset.py        # Statistics for category sample distribution
├── datasets.py               # Dataset loader & customized acoustic data augmentation
├── engine.py                 # Training & evaluation core logic
├── optim_factory.py          # Optimizer wrapper + layer-wise learning rate decay implementation
├── main.py                   # Main training script for HydroacousticNet
├── test.py                   # Test script, output accuracy, F1-score, confusion matrix
├── utils.py                  # Logging, metric calculation, checkpoint saving tools
├── run_with_submitit.py      # Script for cluster GPU training (SLURM)
├── test_results.json         # Sample output of quantitative evaluation metrics
└── models/
    ├── convnext.py           # Customized HydroacousticNet backbone based on ConvNeXt
    └── convnext_isotropic.py # Isotropic ConvNeXt variant for ablation comparison
```

> Note: Temporary runtime files (nohup.out, txt logs) are excluded from the structure for clarity.

---

## 🚀 Quick Start
### Hardware Environment
All experiments are implemented on NVIDIA A100 40G GPU, Python 3.9, PyTorch 1.13, CUDA 11.6.

### 1. Environment Installation
```bash
# Clone this repository
git clone https://github.com/Listen213/gammatone-hydroacousticnet.git
cd gammatone-hydroacousticnet

# Create independent virtual environment (recommended)
conda create -n hydroacoustic python=3.9
conda activate hydroacoustic

# Install all required packages
pip install -r requirements.txt
```

### 2. Dataset Preparation
Download ShipsEar / DeepShip public audio datasets, and organize your audio files and metadata CSV as below:
```
ShipsEar/
├── *.wav
└── shipsEarMeta_90.csv      # Two columns: filename (audio name), type (vessel category)
```
> Important: Modify the root audio path variable inside `full_pipeline.py` to your local dataset folder before running.

### 3. Generate Gammatone Spectrograms & Split Small-Sample Dataset
This script completes audio conversion, fixed train/val/test split (7:1.5:1.5), and outputs the file list of 7% training samples:
```bash
python full_pipeline.py
```
Generated outputs:
- `audio_class_gammatone_images/`: Raw 224×224 Gammatone feature images
- `ships_gammatone_dataset_split/`: Divided train/val/test image folders
- `small_sample_file_list.txt`: Filename list of 7% small training subset

### 4. Train HydroacousticNet Model
```bash
python main.py \
    --data_set image_folder \
    --data_path ships_gammatone_dataset_split/train \
    --eval_data_path ships_gammatone_dataset_split/val \
    --nb_classes 5 \
    --input_size 224 \
    --batch_size 32 \
    --epochs 100 \
    --model hydroacousticnet \
    --layer_decay 0.9 \
    --lr 4e-3 \
    --output_dir ./outputs \
    --imagenet_default_mean_and_std False
```
Key parameter explanation
| Argument | Description |
|----------|-------------|
| `--model hydroacousticnet` | Our customized ConvNeXt-based backbone (not original convnext_tiny) |
| `--layer_decay 0.9` | Exponential layer-wise learning rate decay factor |
| `--nb_classes` | Total classification categories (5 for ShipsEar) |
| `--output_dir` | Path to save training checkpoints and logs |

### 5. Model Evaluation
```bash
# Evaluate full test set, output all quantitative metrics
python test.py
```
The test script automatically calculates overall accuracy, per-class accuracy, F1-score, and saves confusion matrix results into `test_results.json`.

Optional single-image inference:
```bash
python test.py /path/to/your/gammatone_image.png
```

---

## 📊 Quantitative Experimental Results
### ShipsEar Dataset (7% training samples)
| Model Pipeline | Accuracy | F1-score |
|----------------|----------|----------|
| Mel Spectrogram + ResNet-18 | 86.7% | 86.7% |
| Gammatone Filter + ResNet-18 | 87.7% | 87.5% |
| Mel Spectrogram + HydroacousticNet | 91.9% | 91.9% |
| **Ours: Gammatone + HydroacousticNet** | **93.6%** | **93.4%** |

### Cross-Dataset Generalization on DeepShip Subset
| Input Feature | Accuracy | F1-score |
|--------------|----------|----------|
| Mel Spectrogram | 92.5% | 92.4% |
| CQT Spectrogram | 90.8% | 90.6% |
| STFT Spectrogram | 89.0% | 88.9% |
| **Ours Gammatone Filter** | **96.6%** | **96.5%** |

### Per-Class Accuracy on ShipsEar
| Class Label | Vessel Type | Accuracy |
|-------------|-------------|----------|
| A | Fishing & Tug Boats | 97.2% |
| B | Small Motorboats | 88.6% |
| C | Passenger Ships | 87.9% |
| D | Large Ro-Ro / Cargo Ferries | 92.4% |
| E | Marine Ambient Background Noise | 99.5% |

---

## 🔧 Customization Guide
### Train on your own private underwater dataset
1. Use `full_pipeline.py` to convert your audio to Gammatone images
2. Organize feature images into class-separated subfolders
3. Modify `--nb_classes` in training command to match your category count

### Adjust Layer-wise Learning Rate Decay
Change the decay factor via command line argument during training:
```bash
--layer_decay 0.9
```

### Modify Underwater Acoustic Data Augmentation
Edit the transform pipeline function `build_transform()` inside `datasets.py`:
```python
transforms.RandomResizedCrop(args.input_size, scale=(0.85, 1.0)),
transforms.RandomHorizontalFlip(p=0.3),  # Low flip probability to protect temporal sequence
transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
```

---

## 📝 Dataset & Weight Resource Links
All public datasets and pre-trained weights are available at:
1. ShipsEar Dataset: https://github.com/DavidSanAndres/ShipsEar
2. DeepShip Dataset: https://github.com/irfankundi/DeepShip
3. ImageNet-22K pre-trained weights for HydroacousticNet: Auto-downloaded via timm library during model initialization, no manual download required.

---

## 🤝 Contribution Guide
Pull requests and issue discussions are welcome. Please open an issue to discuss major functional improvements before submitting code changes.

---

## 📧 Contact Information
- Qianxing Wei (First Author): 3526016482@qq.com
- Chaochao Sun (Corresponding Author): pop-scc@163.com

All authors: Qianxing Wei, Chaochao Sun, Yuan Peng

---

## 🙏 Acknowledgements
This work is built upon open-source projects:
1. ConvNeXt official implementation from Meta Research
2. ShipsEar & DeepShip public underwater acoustic datasets
3. Librosa audio signal processing library