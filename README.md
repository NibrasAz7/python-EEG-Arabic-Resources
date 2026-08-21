# python-EEG-Arabic-Resources

Companion code repository for the Arabic book **"معالجةُ إشاراتِ تخطيطِ الدماغِ الكهربائيِّ بِبايثون"** (EEG Signal Processing with Python).

All code examples in the book link to files in this repository via `\exlink{path/to/file.py}`.

## Datasets

This repository uses two data sources:

### 1. Local Auditory EEG (PhysioNet)
- **URL**: https://physionet.org/content/auditory-eeg/1.0.0/
- **Author**: Nibras Abu Al-Dhahab
- **Channels**: 4 (P4, Cz, F8, T7)
- **Subjects**: 20
- **Used in**: Chapters 3-9 (signal processing)
- **Download**: See `data/download_local.py`

### 2. MOABB (Mother of All BCI Benchmarks)
- **URL**: https://github.com/NeuroTechX/moabb
- **Datasets**: Motor imagery, P300, SSVEP
- **Used in**: Chapters 11-13 (machine learning)
- **Download**: Automatic via `moabb` Python package

## Setup

```bash
python -m venv eeg_book
source eeg_book/bin/activate  # Linux/macOS
# eeg_book\Scripts\activate   # Windows

pip install -r requirements.txt
python utils/verify_install.py
```

## Repository Structure

```
python-EEG-Arabic-Resources/
├── chapters/           # Python scripts per chapter
│   ├── ch01/           # Chapter 1 scripts
│   ├── ch02/           # Chapter 2 scripts
│   └── ...
├── figures/
│   ├── generated/      # Code-generated figures (PNG/PDF)
│   └── source_book/    # Figures from source book (for reference)
├── utils/
│   ├── eeg_loader.py   # Shared data loading utilities
│   └── verify_install.py
├── data/
│   ├── download_local.py
│   └── README.md
├── requirements.txt
└── README.md
```

## Reproducibility

Every script is designed to be fully reproducible:
- Fixed random seeds
- Explicit data paths
- Documented dependencies
- Clear output expectations
