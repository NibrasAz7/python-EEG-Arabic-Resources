# Data Directory

This directory contains data download scripts. The actual data is NOT stored in the repository.

## Local Auditory EEG

Download from PhysioNet:
```bash
python download_local.py --output ./local_data
```

- **Dataset**: auditory-eeg/1.0.0
- **URL**: https://physionet.org/content/auditory-eeg/1.0.0/
- **Format**: CSV (4 channels: P4, Cz, F8, T7)
- **Size**: ~400 MB (20 subjects, 10 experiments)

## MOABB Datasets

MOABB downloads data automatically when first used:
```python
from moabb.datasets import BNCI2014_001
dataset = BNCI2014_001()
dataset.download()  # Downloads to ~/mne_data
```

- **URL**: https://github.com/NeuroTechX/moabb
- **Datasets used**: BNCI2014-001 (Motor Imagery)
- **Size**: ~2 GB (varies by dataset)
