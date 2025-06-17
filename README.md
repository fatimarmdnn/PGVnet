# PGVnet: Rapid, Physics-Consistent PGV Maps

![PGVnet Pipeline](pgvnet.png)

This repository contains the code and models described in the manuscript:

> **PGVnet: A Machine Learning Framework for the Generation of Rapid, Physics-Consistent PGV Maps**  
> *Submitted to JGR Machine Learning (2025)*


---

##  Installation

```bash
# Clone the repository
git clone https://github.com/fatimarmdnn/PGVnet.git
cd pgvnet

# Create and activate a virtual environment, then install dependencies
python -m venv pgvenv
source pgvenv/bin/activate 
pip install -r requirements.txt
```

---

## Repository Structure

```
pgvnet/
├── data/                 # Contains forward DB, reciprocal DB, trained XGBoost models, and reciever coords
├── src/                  # Source files for training and testing
├── results/              # Checkpoints, predictions, plots, and evaluation metrics
├── Framework_Pipeline.ipynb  # Demonstrates the full pipeline (Steps 1 & 2)
```

---

##  Quickstart: Running the Pipeline

You can run the pipeline either via the notebook `Framework Pipeline.ipynb` or from the command line as shown below.

---

### Step 1: Generate Sparse PGV Maps (XGBoost)

```bash
python src/xgboost_predictor.py --models_dir /media/wolf6819/Elements/models --data_tag 50_50 --spacing_km 4
```

This will:

- Load the trained XGBoost models for each receiver
- Generate sparse PGV predictions to be used as input for the super-resolution model in Step 2

**Required arguments are:**

- `--models_dir`: Path to directory with trained XGBoost models 
- `--data_tag`: (e.g. 50_50 = 50 locations × 50 samples)
- `--spacing_km`: Grid spacing in km (4, 6, or 8)

---

### Step 2: Train the Super-Resolution Model

```bash
python src/encoderMLP_predictor.py --mode train --data_tag 50_50_x4 --downsample_factor 4 
```

This will:

- Load and preprocess the sparse maps dataset
- Train the encoderMLP network
- Save model checkpoints and learning curves to `results/`

**Required arguments are:**

- `--mode`: Choose from `train`, `test`, or `inference`
- `--data_tag`: (e.g. 50_50_x4 = 50 locations × 50 samples, downsampled ×4)
- `--downample_factor`: Downsampling factor to apply (e.g. 4)

---

### Step 3: Evaluate the Model

```bash
python src/encoderMLP_predictor.py --mode test --data_tag 50_50_x4 --downsample_factor 4 --results_dir ./results/20250617_091441_28edf9
```

- Loads the best saved checkpoint
- Evaluates model on the test set
- Outputs reconstructed PGV fields and performance metrics

**Required arguments are:**

- `--mode`,  `--data_tag`, `--downample_factor`
- `--results_dir`: Path to the directory containing the trained model checkpoints

---

## Outputs

The model outputs and logs are saved under `results/<run_id>/`. Key files include:

- `best_model.pth`          : Trained model weights
- `test_map_sim*_comp*.png` : True vs Predicted PGV maps of an example event along component 0 (East) or 1 (North)
- `test_preds.npy`, `test_gts.npy`: NumPy dumps of the true and predicted pgv maps
- `learning_curves.png`: Training loss/metric curves
- `test_metrics.txt`   : Quantitative evaluation metrics

---

## Acknowledgements

Some code components were adapted from the following repository:


- [maxjiang93/space_time_pde](https://github.com/maxjiang93/space_time_pde)

---

## Citation

If you use this repository in your work, please cite:

```
F. Ramadan, T. Nissen-Meyer, P. Koelemeijer, B. Fry (2025). 
PGVnet: A Machine Learning Framework for the Generation of Rapid, Physics-Consistent PGV Maps.
```