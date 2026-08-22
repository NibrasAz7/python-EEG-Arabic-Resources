# python-EEG-Arabic-Resources

> Companion code repository for the Arabic book **"معالجةُ إشاراتِ تخطيطِ الدماغِ الكهربائيِّ بِبايثون"** (EEG Signal Processing with Python: Machine Learning Techniques for Brain-Computer Interface Development).

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-green?logo=scipy&logoColor=white)](https://scipy.org/)
[![MNE](https://img.shields.io/badge/MNE--Python-1.5+-red?logo=python&logoColor=white)](https://mne.tools/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-orange?logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![Dataset](https://img.shields.io/badge/Dataset-PhysioNet-blueviolet)](https://physionet.org/content/auditory-eeg/1.0.0/)
[![Book](https://img.shields.io/badge/Book-Arabic%20LaTeX-success)](https://github.com/NibrasAz7/python-for-EEG-Arabic)

---

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Repository Structure](#repository-structure)
- [Chapter Progress](#chapter-progress)
- [Setup](#setup)
- [Usage](#usage)
- [Colab Notebooks](#colab-notebooks)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Author](#author)

---

## Overview

This repository contains all Python scripts and Jupyter notebooks referenced in the book. Every code listing in the book has:

- A standalone `.py` script (linked via `\exlink{}`)
- A bilingual Colab notebook: `_ar.ipynb` (Arabic markdown) and `_en.ipynb` (English markdown)
- The book's "Open in Colab" button links to the `_ar.ipynb` version

The code uses a real EEG dataset from PhysioNet, not synthetic data, so students work with authentic brain signals from the first chapter.

---

## Datasets

### 1. Auditory Evoked Potential EEG-Biometric Dataset (PhysioNet)

| Property | Value |
| -------- | ----- |
| URL | <https://physionet.org/content/auditory-eeg/1.0.0/> |
| Author | Nibras Abo Alzahab et al. (2021) |
| Format | WFDB (`.dat` + `.hea`) |
| Channels | 4 (P4, Cz, F8, T7) |
| Sampling rate | 200 Hz |
| Subjects | 20 |
| Experiments | 10 per subject |
| Used in | Chapters 3-9 (signal processing) |
| Download | `python data/download_local.py --output data/local --subjects 1` |

### 2. MOABB (Mother of All BCI Benchmarks)

| Property | Value |
| -------- | ----- |
| URL | <https://github.com/NeuroTechX/moabb> |
| Datasets | Motor imagery, P300, SSVEP |
| Used in | Chapters 11-13 (machine learning) |
| Download | Automatic via `moabb` Python package |

---

## Repository Structure

```text
python-EEG-Arabic-Resources/
├── chapters/                # Python scripts + notebooks per chapter
│   ├── ch01/                # Introduction
│   ├── ch02/                # EEG fundamentals
│   ├── ch03/                # Data loading & visualization
│   ├── ch04/                # Bandpass filters (highpass, lowpass, bandpass)
│   ├── ch05/                # Smoothing filters
│   ├── ch06/                # Notch filters
│   ├── ch07/                # Artifact removal
│   ├── ch08/                # Real-time streaming (LSL)
│   ├── ch09/                # Feature extraction
│   ├── ch10/                # Time-frequency analysis
│   ├── ch11/                # BCI paradigms (MI, P300, SSVEP, ErrP)
│   ├── ch12/                # Classical ML classifiers
│   ├── ch13/                # Evaluation metrics
│   ├── ch14/                # Deep learning intro
│   └── ch15/                # EEGNet & transfer learning
├── data/
│   ├── download_local.py    # PhysioNet dataset downloader
│   └── README.md            # Dataset documentation
├── figures/
│   ├── generated/           # Code-generated figures (PNG/PDF)
│   └── source_book/         # Figures from source book (reference)
├── utils/
│   ├── eeg_loader.py        # Shared EEG loading utility (WFDB)
│   └── verify_install.py    # Installation verification
├── requirements.txt
└── README.md
```

---

## Chapter Progress

| Chapter | Topic | Scripts | Notebooks | Status |
| ------- | ----- | ------- | --------- | ------ |
| 1 | Introduction | | | Planned |
| 2 | EEG fundamentals | | | Planned |
| 3 | Data loading & visualization | | | Planned |
| 4 | Bandpass filters | 3 | 6 (3 AR + 3 EN) | Complete |
| 5 | Smoothing filters | | | Planned |
| 6 | Notch filters | | | Planned |
| 7 | Artifact removal | | | Planned |
| 8 | Real-time streaming (LSL) | | | Planned |
| 9 | Feature extraction | | | Planned |
| 10 | Time-frequency analysis | | | Planned |
| 11 | BCI paradigms | | | Planned |
| 12 | Classical ML classifiers | | | Planned |
| 13 | Evaluation metrics | | | Planned |
| 14 | Deep learning intro | | | Planned |
| 15 | EEGNet & transfer learning | | | Planned |

---

## Setup

### Local installation

```bash
python -m venv eeg_book
source eeg_book/bin/activate  # Linux/macOS
# eeg_book\Scripts\activate   # Windows

pip install -r requirements.txt
python utils/verify_install.py
```

### Download the dataset (one subject for quick testing)

```bash
python data/download_local.py --output data/local --subjects 1
```

### Download the full dataset (all 20 subjects)

```bash
python data/download_local.py --output data/local
```

---

## Usage

### Running a script locally

```bash
cd python-EEG-Arabic-Resources
python chapters/ch04/highpass.py
```

### Opening in Colab

Each chapter's notebooks are available in two languages:

- `chapters/chXX/name_ar.ipynb` — Arabic markdown (linked from the book)
- `chapters/chXX/name_en.ipynb` — English markdown

To open in Colab, replace `github.com` with `colab.research.google.com/github`:

```text
https://colab.research.google.com/github/NibrasAz7/python-EEG-Arabic-Resources/blob/main/chapters/ch04/highpass_ar.ipynb
```

---

## Colab Notebooks

Every notebook is self-contained and follows this structure:

1. Install dependencies (`pip install`)
2. Clone this repository
3. Download one subject's data (fast for educational purposes)
4. Load and process the EEG signal
5. Interactive plotly visualization (zoom, pan, hover)
6. Summary of key takeaways

Arabic notebooks use `<div dir="rtl">` tags for proper right-to-left rendering in Colab.

---

## Reproducibility

Every script is designed to be fully reproducible:

- Fixed random seeds
- Explicit data paths
- Documented dependencies
- Clear output expectations
- Sampling rate fixed at 200 Hz (matching the PhysioNet dataset)
- `load_local_eeg()` utility ensures consistent data loading across all chapters

---

## Citation

If you use this dataset or code, please cite:

```text
Abo Alzahab, N., et al. (2021). Auditory Evoked Potential EEG-Biometric Dataset.
PhysioNet. https://physionet.org/content/auditory-eeg/1.0.0/
```

---

## Author

**Dr. Nibras Abo Alzahab**

- NIBRAS Center for BCI and Neurotechnology

---

## License

This project is licensed under the MIT License.
