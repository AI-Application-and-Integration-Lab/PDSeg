# 🔬 PDSeg: Patch-Wise Distillation and Controllable Image Generation for Weakly-Supervised Histopathology Tissue Segmentation
[![Paper](https://img.shields.io/badge/Paper-ICASSP25-blue)](https://ieeexplore.ieee.org/abstract/document/10888097) [![License](https://img.shields.io/badge/license-custom-lightgrey)](./LICENSE)

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
python 3.11
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

## 📁 Datasets and Our checkpoints
You can get the LUAD-HistoSeg and BCSS-WSSS dataset from [here](https://github.com/ChuHan89/WSSS-Tissue) and our [checkpoints](https://gofile.me/5R0b8/Is9Jufem1).

If your dataset is in a different folder, make a soft-link from the target dataset to the `data` folder. We expect the following tree:
```
data/
    BCSS-WSSS/
      training/
      test/
      val/
    LUAD-HistoSeg/
checkpoint/
    bcss_baseline/
      best_model.pth
    luad_baseline/
      best_model.pth
... other files 
```
## 🏋️ Training and Evaluation

### for LUAD-HistoSeg
```
bash run_luad.sh
```

### for BCSS-WSSS
```
bash run_bcss.sh
```

## 📝 Citation

If you find this work useful in your research, please cite our paper:

```
@inproceedings{li2025pdseg,
  title={PDSeg: Patch-Wise Distillation and Controllable Image Generation for Weakly-Supervised Histopathology Tissue Segmentation},
  author={Li, Wei-Hua and Hsieh, Yu-Hsing and Yang, Huei-Fang and Chen, Chu-Song},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## 📄 License
This project is released under a custom license.  
Please see the [LICENSE](./LICENSE) file for the full terms and conditions.

For academic or commercial use, please contact the authors.

---
Stay tuned for code release and updates!  
📬 Questions or feedback? Feel free to open an issue or contact us.
