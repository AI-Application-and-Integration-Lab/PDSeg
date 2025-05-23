# 🔬 PDSeg: Patch-Wise Distillation and Controllable Image Generation for Weakly-Supervised Histopathology Tissue Segmentation
[![Paper](https://img.shields.io/badge/Paper-ICASSP25-blue)](https://ieeexplore.ieee.org/abstract/document/10888097)

Welcome! 👋 This repository contains the official implementation of our **ICASSP 2025** paper:  
**"PDSeg: Patch-Wise Distillation and Controllable Image Generation for Weakly-Supervised Histopathology Tissue Segmentation"**.

We propose a novel framework that combines patch-wise knowledge distillation with controllable image generation to push the boundaries of weakly-supervised tissue segmentation in histopathology images.

---

## 🖼️ Overview

<p align="center">
  <img src="image/pdseg.png" width="800" alt="PDSeg Overview"/>
</p>

---

<details>
  <summary>📊 <b>Click to view: Comparison on LUAD-HistoSeg and BCSS-WSSS</b></summary>

  <br>

  <p align="center">
    <img src="image/comparison.png" width="800" alt="Comparison on datasets"/>
  </p>

</details>

---

## ⚙️ Installation and Requirements
We are tested under:
```
python 3.8
```
If you want to install a custom environment for this code, you can run the following using [conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html):
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install matplotlib

pip install timm==0.6.13
pip install opencv-python
pip install einops
pip install scikit-learn
pip install scikit-image
```

## 📁 Datasets
You can get the LUAD-HistoSeg and BCSS-WSSS dataset from [here](https://github.com/ChuHan89/WSSS-Tissue).

If your dataset is in a different folder, make a soft-link from the target dataset to the `data` folder. We expect the following tree:
```
dataset/
data/
    BCSS-WSSS/

... other files 
```
## 🏋️ Training

```
bash run_bcss.sh
```

## 🔍 Inference

---
Stay tuned for code release and updates!  
📬 Questions or feedback? Feel free to open an issue or contact us.
