## Introduction
This repository contains several materials that supplements the Spoofing-Aware Speaker Verification (SASV) Challenge 2022 including:
- calculating metrics;
- extracting speaker/spoofing embeddings from pre-trained models;
- training/evaluating Baseline2 in the evaluation plan. 

More information can be found in the [webpage](https://sasv-challenge.github.io) and the [evaluation plan](pdfs/2022_SASV_evaluation_plan_v0.2.pdf) 

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
Extracted embeddings will be saved in `./embeddings`.
- Speaker embeddings are extracted using the ECAPA-TDNN [3].
  - Implmented by https://github.com/TaoRuijie/ECAPATDNN
- Spoofing embeddings are extracted using the AASIST [2].
- We also prepared extracted embeddings.
  - To use prepared emebddings, git-lfs is required. Please refer to [https://git-lfs.github.com](https://git-lfs.github.com) for further instruction. After installing git-lfs use following command to download the embeddings.
    ```
    git-lfs install
    git-lfs pull
    ```


```
python save_embeddings.py
```

## Baseline 2 Training
Run below script to train Baseline2 in the evaluation plan.
- It will reproduce **Baseline2** described in the Evaluation plan.
```
python main.py --config ./configs/baseline2.conf
```

## Developing own models
- Currently adding...

### Adding custom DNN architecture
1. create new file under `./models/`.
2. create a new configuration file under `./configs`
3. in the new configuration, modify `model_arch` and add required arguments in
`model_config`.
4. run `python main.py --config {USER_CONFIG_FILE}` 
### Using only metrics
Use `get_all_EERs` in `metrics.py` to calculate all three EERs.
- prediction scores and keys should be passed on using 
  - `protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt` or
  -  `protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt`

## References
[1] ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech
```bibtex
@article{wang2020asvspoof,
  title={ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech},
  author={Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and Delgado, H{\'e}ctor and Nautsch, Andreas and Evans, Nicholas and Sahidullah, Md and Vestman, Ville and Kinnunen, Tomi and Lee, Kong Aik and others},
  journal={Computer Speech \& Language},
  volume={64},
  pages={101114},
  year={2020},
  publisher={Elsevier}
}
```
[2] AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
```bibtex
@inproceedings{Jung2022AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={Proc. ICASSP}, 
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks}, 
  year={2022}
```
[3] ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification
```bibtex
@inproceedings{desplanques2020ecapa,
  title={{ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification}},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={Proc. Interspeech 2020},
  pages={3830--3834},
  year={2020}
}
```
