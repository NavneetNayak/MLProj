# Cox Proportional Hazards Model for Pancreatic Adenocarcinoma Survival Prediction

Using the TCGA-PAAD dataset, The model predicts relative risk (hazard ratios) for patients based on clinical and genomic features.

## How to Run?
Install Requirements:
```
pip3 install -r requirements.txt
```

If using nix:
```
nix develop
```

### Inference
Demo Inference: Picks 5 random samples from the dataset to run inference. (pre-trained model is already present in `bin/`)
```
python3 inference.py
```

### Training
To see the training process or re-train the model, run `training.ipynb`


