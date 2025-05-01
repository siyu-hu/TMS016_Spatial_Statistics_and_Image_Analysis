# TMS016_Spatial_Statistics_and_Image_Analysis
Assignments 2025


## Project 3 Fingerprint-Siamese (FVC2000)  🖐️🔍

A minimal solution completed in a short time for **fingerprint-matching** on the public **[FVC 2000 / DB1_B](http://bias.csr.unibo.it/fvc2000/databases.asp)** subset. The model is a Siamese CNN trained *only* on DB1_B; given any two DB1_B `.tif` images, it decides whether they originate from the **same finger**.

> **Note on cross-database generalisation**  
> We experimented with fine-tuning / zero-shot transfer to **DB2_B or DB3_B**, but the results were unsatisfactory. Consequently, the current code base does not include cross-subset migration utilities.

``data/`` and ``checkpoints/`` are intentionally **git-ignored**. 

| Stage | Script | Main outputs |
|-------|--------|--------------|
| 0️⃣  Preprocessing | `preprocess.py` | • Enhanced **.tif** (for inspection)<br>• Normalised **.npy** (network input) |
| 1️⃣  Pair generation | `create_train_pairs.py`  | Train / val **pair lists** |
| 2️⃣  Training | `train.py` | Model checkpoints (`checkpoints/best.pt`) |
| 3️⃣  Validation | `validate.py` | ROC / metrics vs thresholds /   |
| 4️⃣  Batch inference | `inference_batch.py` | Match-score CSV, visualisations |
| 🔄  Data augmentation (optional) | `augment_images.py` | On-the-fly augmentations for stage 1, Apply random affine transformation (small rotation + scaling) to a single image.  |


### Methods and Results

methods

...


Classification Report:
  Accuracy  : 97.50%
  Precision : 0.9902
  Recall    : 0.7250
   F1 Score  : 0.8371
  TP=203 | TN=2878 | FP=2 | FN=77
Inference Total pairs: 3160

--- 

### Step-by-step

### 1. Environment

```bash
# Recommended: Python ≥3.9 + venv / conda
pip install -r requirements.txt
```
or

```bash
# Recommended: Python ≥3.9 + venv / conda
conda env create -f environment.yml  
conda activate fvc-fingerprint
cd project3_fingerprint_fvc2000
```


### 2. Folder layout (after you run everything once)

```
project-root/
│
├─ data/
│   ├─ original/                # raw + enhanced images
│   │   ├─ DB1_B/               # raw .tif from the FVC-2000 DB1_B set
│   │   └─ DB1_B_new/           # enhanced .tif created by preprocess.py
│   ├─ processed/               # network-ready .npy files
│   │   └─ DB1_B_new_1/
│   └─ pairs files (*.npz)      # training / val pairs 
│
├─ checkpoints/                 # *.pth saved by train.py
├─ outputs/                     # simple visulization for training and validation loss / metrics
└─ *.py                         # source code
```

### 3. Pre-process the raw fingerprints

```bash
python preprocess.py \
  --input   ./data/original/DB1_B \
  --out-npy ./data/processed/DB1_B_new_1 \
  --out-tif ./data/original/DB1_B_new_1 \
  --size    300
```

*Advanced pipeline:* Gamma → zero-mean/unit-var normalisation → orientation field →
overlapping-block Gabor → CLAHE → resize.

The resulting **.npy** files are `float32`, shape `(300, 300)`, range `0–1`.

### 4. Build training / validation pairs

```
python create_train_pairs.py
```

### 5. Train

```bash
python train.py \
  --train_pairs ./data/*.npz \
  --val_pairs ./data/*.npz \
  --use_aug \
  --balance_neg \
  --finetune \
  --batch_size \
  --lr 1e-3
  
```
if finetune on certain checkpoints: 

```bash
python train.py \
  --train_pairs ./data/*.npz \
  --val_pairs ./data/*.npz \
  --use_aug \
  --balance_neg \
  --finetune \
  --best_ckpt ./checkpoints/*.pt 
  
```

### 6. Validate

```bash
python validate.py \
  --val_data ./data/*.npz \ 
  --ckpt ./checkpoints/*.pt \
  --threshold 1.01
```

### 7. Run batch inference on unseen data

With 12% Sample calibration

```bash
python inference_batch.py \
  --inference_data ./data/original/DB1_B \
  --ckpt ./checkpoints/model.pt \
  --auto_threshold
```

---

* **Changing the CNN architecture** – edit `siamese_model.py`; keep the output
  embedding dimension consistent across training and inference.
* **Different FVC subsets** – simply pass a different `--input` folder to
  `preprocess.py`; pair-generation is fully data-driven.
* **GPU vs. CPU** – the heavy lifting is in PyTorch; OpenCV Gabor kernels run on
  CPU. If preprocessing becomes a bottleneck, experiment with smaller
  `block_size` or multiprocessing.
* **Reproducibility** – `train.py` seeds `torch`, `numpy`, and Python’s built-in
  RNG; pass `--deterministic` for deterministic CuDNN.

