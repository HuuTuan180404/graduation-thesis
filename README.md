# LSA64 Experiments

This repository contains training experiments for the LSA64 dataset.

```
@inproceedings{ronchetti2016lsa64,
    title={LSA64: an Argentinian sign language dataset},
    author={Ronchetti, Franco and Quiroga, Facundo and Estrebou, C{\'e}sar Armando and Lanzarini, Laura Cristina and Rosete, Alejandro},
    booktitle={XXII Congreso Argentino de Ciencias de la Computaci{\'o}n (CACIC 2016).},
    year={2016}
}
```

---

## Dataset

For the experiments in this project, we use datasets as provided in the official repository of the Siformer model.

To run this project, you need to download the dataset from the following link:

**Dataset download link:** [LSA64 Dataset](https://drive.google.com/drive/folders/133oQqcp_4BZU8u8mR2R0_0tthDG5P7t5)

Expected project structure:

```text
project_root/
│
├── datasets/
│   ├── czech_slr_dataset.py
│   └── LSA64_60fps.csv
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
python train_LSA64.py
```
