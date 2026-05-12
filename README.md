# WLASL100 Experiments

This repository contains training experiments for the WLASL100 dataset.

```
@inproceedings{li2020word,
    title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
    author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
    booktitle={The IEEE Winter Conference on Applications of Computer Vision},
    pages={1459--1469},
    year={2020}
}
```

---

## Dataset

For the experiments in this project, we use datasets as provided in the official repository of the Siformer model.

To run this project, you need to download the dataset from the following link:

**Dataset download link:** [WLASL100 Dataset](https://drive.google.com/drive/folders/1zbiN-mjkWEk_cVrpiEHGAXWI4lIwg4tF)

Expected project structure:

```text
project_root/
│
├── datasets/
│   ├── czech_slr_dataset.py
│   ├── WLASL100_train_25fps.csv
│   └── WLASL100_val_25fps.csv
│
├── ...
```

---

## Environment Setup

This project uses a Conda environment defined in `environment.yml`.

### 1. Create the environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate <environment_name>
```

---

## Training

```bash
python train_WLASL100.py
```
