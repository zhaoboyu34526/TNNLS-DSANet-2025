# Domain Information Mining and State-Guided Adaptation Network (DSAnet)

> **Domain Information Mining and State-Guided Adaptation Network for Multispectral Image Segmentation**  
> *Boyu Zhao, Mengmeng Zhang, Wei Li*, Yunhao Gao, Junjie Wang*  
> IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2025
https://ieeexplore.ieee.org/document/11089978/

---

## 🧠 Framework Overview

<p align="center">
  <img src="./Frame.jpg" width="80%">
</p>

---

## 🧩 Abstract

**Segment Anything (SAM)**, as a prompt-based image segmentation foundation model, demonstrates strong task versatility and domain generalization capabilities, providing a new direction for cross-scene segmentation.  
However, SAM still faces challenges in **multi-spectral cross-domain segmentation**, mainly reflected in:

1. **Limited domain information utilization** — For the **source domain (SD)**, SAM is primarily trained on visible light images, causing its features to focus mainly on RGB semantics while neglecting non-visible spectral information. For the **target domain (TD)**, SAM overlooks the spectral and spatial domain shift between SD and TD, as it is fine-tuned using only labeled SD samples and ignores unlabeled TD data.  
2. **Limited cross-domain adaptation capability** — Although SAM learns certain domain-invariant features from large-scale pretraining, its structure lacks explicit domain adaptation mechanisms. When SD and TD exhibit significant distribution differences, SAM struggles to adapt directly to TD data, leading to degraded segmentation performance.

To address these challenges, we combine the strengths of **Masked Autoencoder (MAE)** and **cross-domain adaptation** to propose **DSAnet** (*Domain Information Mining and State-Guided Adaptation Network*), an improved SAM-based framework for multi-spectral segmentation.

- At the **data level**, DSAnet introduces a **Style Masking Learning** module that randomly masks image features and replaces them with domain-specific learnable tokens. This mechanism, integrated with reconstruction tasks, mines both domain-invariant and style-specific representations.  
- At the **task level**, DSAnet employs **Domain State Learning** and **Style-Guided Segmentation**:
  - *Domain State Learning* models inter-domain differences through learnable state sequences for SD and TD, mitigating task shift and enabling direct inference usage.  
  - *Style-Guided Segmentation* leverages TD style prompts to guide SD segmentation training, enhancing SAM’s adaptability to multi-spectral cross-domain tasks.

Extensive experiments on two multi-temporal MSI datasets demonstrate the superiority of DSAnet compared with state-of-the-art cross-domain and SAM-based methods.

---

## ⚙️ Environment Setup

The environment configuration follows the setup in [**MobileSAM**](https://github.com/ChaoningZhang/MobileSAM).  
Please refer to that repository for detailed dependency installation and environment preparation.

---

## 🧪 Experimental Steps

### (1) Data Preparation and Cropping

Prepare datasets and crop them into **512×512** image patches for training and validation.

- **GID Dataset:** Download from [https://captain-whu.github.io/GID/](https://captain-whu.github.io/GID/)  
- **Yellow River Dataset:** Download from [https://github.com/zhaoboyu34526/Alliance](https://github.com/zhaoboyu34526/Alliance)

After downloading, modify the dataset paths in `data_cut.py`, then execute:

```bash
python data_cut.py  
```

### (1) Path Configuration

Modify the dataset path arguments in `train_adapttest.py`:

```bash
parser.add_argument('--train_map',   default=r'/zbssd/yuyu/code/data512/experiment2021/train/img/',   type=str)
parser.add_argument('--train_label', default=r'/zbssd/yuyu/code/data512/experiment2021/train/label/', type=str)
parser.add_argument('--val_map',     default=r'/zbssd/yuyu/code/data512/experiment2021/val/img/',     type=str)
parser.add_argument('--val_label',   default=r'/zbssd/yuyu/code/data512/experiment2021/val/label/',   type=str)
```
### (1) Model Training

Run the following command to start training:

```bash
python train_adapttest.py
```
The training logs and checkpoints will be saved automatically under the default experiment directory.

## 📊 Run Comparison Methods

All baseline and comparison methods can be found in the `compare/` directory.
Each method has a corresponding `.py` file that can be directly executed for independent experiments.
```bash
cd compare
python compare_xxx.py
```

## 📚 Citation

If you find this work useful, please cite:
```bash
@ARTICLE{zhao2025dsanet,
  author={Zhao, Boyu and Zhang, Mengmeng and Li, Wei and Gao, Yunhao and Wang, Junjie},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Domain Information Mining and State-Guided Adaptation Network for Multispectral Image Segmentation}, 
  year={2025},
  volume={36},
  number={11},
  pages={19849-19863}
}
```
