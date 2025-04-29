# Semantic Segmentation on the Mapillary Dataset

Semantic segmentation of street‐level imagery using DeepLabV3 and U‑Net architectures on the Mapillary Vistas dataset.

---

## 📋 Table of Contents

- [Overview](##Overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Preparation and Visualization](#dataset-preparation)
- [Training](#training)
- [Model Architectures](#model-architectures)
- [Utilities](#utilities)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🚀 Overview

This repository provides end‑to‑end code for semantic segmentation of urban scenes using the Mapillary Vistas dataset. It includes implementations of:

- **DeepLabV3** with ResNet backbone and ASPP module
- **U‑Net** model for semantic segmentation

You can train, evaluate, and run inference on your own images, and compute metrics such as Dice score. You can also experiment with Wandb.

I have written a comprehensive report of the architectures used and the result of the training and evaluation. Please, visit the published article here: [My journey into Semantic Segmentation](https://medium.com/@uzohchinedu/deep-learning-for-computer-vision-my-journey-into-semantic-segmentation-45b9d0491d0f).  

<!-- ## ✨ Features

- Dataset loader for Mapillary Vistas (`mapillary_dataset.py`)
- Pretrained ResNet backbone support
- DeepLabV3 and U‑Net model implementations
- Training loop with customizable losses and metrics
- Evaluation scripts to compute Dice and other segmentation metrics -->

## 🗂️ Repository Structure

```
mapillary-semantic-segmentation/
├── data                               # Create this folder with Mapillary data, download from https://www.mapillary.com/dataset/vistas
├── notebooks/                         # Jupyter notebooks for analysis and visualization
│   ├── basic_stats.ipynb              # Exploratory statistics notebook to obtain image class distributions
│   ├── class_distribution_v2.0.json   # Class distribution output data
│   ├── deeplab_test.ipynb             # Notebook for testing DeepLab model
│   └── visualize.ipynb                # Notebook for visualizing predictions in the case of UNET
├── src/
│   ├── dataset/
│   │   └── mapillary_dataset.py      # Data loader for Mapillary data
│   ├── models/
│   │   └── deeplabv3_model/           # Model implementations
│   │       ├── pretrained_resnet/     # Pretrained ResNet weights
│   │       │   ├── resnet18-5c106cde.pth
│   │       │   ├── resnet34-333f7ec4.pth
│   │       │   └── resnet50-19c8e357.pth
│   │       │  
│   │       ├── aspp.py                # Atrous Spatial Pyramid Pooling
│   │       ├── deeplabv3.py           # DeepLabV3 model definition. Note! pretrained weight is called here. 
│   │       ├── resnet.py              # ResNet backbone definition
│   │       ├── unet_model.py          # U-Net model definition
│   │       └── unet_parts.py          # U-Net encoder/decoder blocks
│   │      
│   └── utils/
│       ├── dice_score.py             # Dice coefficient implementation
│       ├── losses_and_metrics.py     # Loss functions and metrics
│       ├── evaluate.py               # Evaluation script
│       └── train.py                  # Training script, main file
└── README.md                         # (this file)
```

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mapillar-semantic-segmentation.git
   cd mapillary-semantic-segmentation/src
   ```

2. Create a Python environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install \
   torch \  # PyTorch: tensor operations & deep learning framework
   torchvision \  # Torchvision: datasets, models, and transforms for computer vision
   matplotlib \  # Matplotlib: plotting library for creating static visualizations
   jupyter \  # Jupyter: interactive notebook environment
   albumentations \  # Albumentations: fast image augmentation for images
   plotly \  # Plotly: interactive, web-based visualizations
   pillow \  # Pillow: image processing library
   numpy \  # NumPy: fundamental package for numerical computing
   tqdm      # TQDM: progress bars for iterables
   ```


## 🗄️ Dataset Preparation and Visualization

1. **Download** the Mapillary Vistas dataset following the [official instructions](https://www.mapillary.com/dataset/vistas).
2. **Inspect and run** the notebook `basic_stats.ipynb` to obtain basic statatics like the size of the images and the class distribution.
     

## 🏋️ Training

Train a deeplabv3 or unet model :
```bash
python train.py \
  --model deeplabv3 \
  --batch_size 4 \
  --epochs 20 \
```

- Default loss: CrossEntropy + Dice loss (see `losses_and_metrics.py`)
- Checkpoints and logs will be saved under `--output_dir`.



Results will be printed to console and saved as JSON in the checkpoint folder.

## 🧩 Model Architectures

- **DeepLabV3**: Uses a ResNet encoder and ASPP module for multi‑scale context aggregation.
- **U‑Net**: Classic encoder‑decoder with skip connections.

Model code lives under `models/deeplabv3_model/` and at the root of `models/` for U‑Net.

## 🔧 Utilities

- **`dice_score.py`**: Compute Dice coefficient.
- **`losses_and_metrics.py`**: Built‑in loss functions (CrossEntropy, Dice, Focal) and metrics.
- **`train.py`**: End‑to‑end training loop with logging and wandb experimentation.
- **`evaluate.py`**: Compute and report metrics on a dataset.

## 🤝 Contributing

Contributions are welcome! 

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## ✉️ Contact

Feel free to open issues or reach out with questions and suggestions!
