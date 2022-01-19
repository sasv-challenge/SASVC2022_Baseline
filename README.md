### Data preparation
```
python ./aasist/download_dataset.py

mkdir -p data/_meta/
cp ./LA/*protocols/*.txt ./data/_meta/


python save_embeddings.py
```

### Training
```
python main.py
```