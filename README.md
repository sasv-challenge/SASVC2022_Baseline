## Introduction
This repository contains several materials that supplements the Spoofing-Aware Speaker Verification (SASV) Challenge 2022 including:
- calculating metrics;
- extracting speaker/spoofing embeddings from pre-trained models;
- training/evaluating Baseline2 in the evaluation plan. 

### Prerequisites
#### Load ECAPA-TDNN & AASIST repositories
```
git submodule init
git submodule update
```

#### Install requirements
```
pip install -r requirements.txt
```
### Data preparation
The ASVspoof2019 LA dataset [1] can be downloaded using the scipt in AASIST [2] repository
```
python ./aasist/download_dataset.py
```

### Speaker & spoofing embedding extraction
Speaker embeddings and spoofing embeddings can be extracted using below script.

```
python save_embeddings.py
```

### Baseline 2 Training
Run below script to train Baseline2 in the evaluation plan.
```
python main.py
```
