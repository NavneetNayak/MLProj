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
Demo Inference: Picks 5 random samples from the dataset to run inference.
```
python3 inference.py
```

### Training
Run the `.ipynb` file to train the model.


