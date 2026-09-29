# Pneumonia Detection — An MLOps Pipeline

A chest X-ray classifier built end-to-end: data validation, transfer learning, a Flask API with a live demo frontend, Docker containerization, and an automated CI pipeline on GitHub Actions.

This project was built as a practical application of concepts studied in *Practical MLOps* — each stage below maps directly to something the book teaches.

---

## Problem Statement

**Task:** Given a chest X-ray image, classify it as **NORMAL** or **PNEUMONIA** — a binary image classification problem.

- The model outputs a probability between 0 and 1.
- A threshold of **0.5** converts that probability into a label: `> 0.5` → PNEUMONIA, `≤ 0.5` → NORMAL.
- **Why not just optimize for accuracy:** the dataset is imbalanced (PNEUMONIA outnumbers NORMAL roughly 3:1), so accuracy alone can hide a model that never really learned anything. We track **precision, recall, and F1** instead, with **recall on PNEUMONIA** as the priority metric — missing a real case (a false negative) is far more costly than a false alarm a doctor can rule out.

---

## Dataset

**Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney (Kaggle), licensed CC BY 4.0.

- 5,863 images total, pre-split into `train/` and `test/` folders.
- Each split is organized into `NORMAL/` and `PNEUMONIA/` subfolders.
- Train: 5,216 images (1,341 NORMAL, 3,875 PNEUMONIA)
- Test: 624 images (234 NORMAL, 390 PNEUMONIA)
- The dataset's provided `val/` folder has only 16 images — too small to be reliable, so a validation split is instead carved out of `train/` (80/20) during training.

### Data validation

Before training, `src/validate_data.py` is run to catch problems early — the book's DataOps lesson applied directly:
- Opens every image and checks it isn't corrupt or unreadable.
- Reports the real class counts and imbalance ratio per split.
- Reports image format and size ranges, confirming resizing is needed before training.

```bash
python3 src/validate_data.py
```

---

## Preprocessing

Applied automatically as images load, via `ImageDataGenerator`:

| Step | What it does | Why |
|---|---|---|
| **Resize** | Every image → 224×224 | Matches MobileNetV2's required input size |
| **Normalize** | Pixel values scaled from 0–255 to 0–1 | Neural networks train far better on small numbers |
| **Augment** | Small random rotations, zooms, flips (training set only) | Improves generalization, partly counters class imbalance |
| **Validation split** | 20% of `train/` held out | The dataset's own `val/` folder is too small to trust |

---

## Model

**Architecture:** MobileNetV2, pretrained on ImageNet (1.4 million images), with transfer learning.

**Why MobileNetV2, and not a heavier model:**
- The dataset is relatively small (~5,200 training images) — training a CNN from scratch would likely overfit.
- MobileNetV2 is lightweight, fast on CPU, and edge-friendly — a good fit for limited hardware and for later containerized deployment.
- A heavier network (VGG16, ResNet50) would cost more to train and deploy, with little accuracy benefit at this dataset size.

**How it's used:**
- MobileNetV2's pretrained layers are **frozen** (not retrained).
- A small new classification head is added on top: `GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)`.
- Only this new head is trained — the model borrows MobileNetV2's existing visual knowledge instead of learning from zero.

**Training configuration:**
- 10 epochs, batch size 32
- **Class weighting** applied to counter the ~3:1 imbalance — the model is told to pay more attention to mistakes on the minority NORMAL class
- **Early stopping** on validation loss, restoring the best-performing weights automatically

```bash
python3 src/train.py
```

---

## Results

Measured on the 624 held-out test images (never seen during training):

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| NORMAL | 0.96 | 0.73 | 0.83 |
| PNEUMONIA | 0.86 | 0.98 | 0.91 |

**Overall accuracy: 88%** — but recall is the number that matters most here: **382 of 390 real pneumonia cases were caught, with only 8 missed.**

Confusion matrix:
```
                 Predicted NORMAL   Predicted PNEUMONIA
Actual NORMAL         170                  64
Actual PNEUMONIA        8                 382
```

---

## Serving: Flask API + Frontend

A trained model file isn't usable on its own — it needs an API in front of it, and the exact same preprocessing used in training must repeat at serving time (the training/serving skew problem).

**`src/app.py`** exposes:
- `GET /` — a browser-based demo page: drag-and-drop an X-ray, see a live prediction with a confidence meter
- `POST /predict` — accepts an image file, returns JSON: `{"prediction": "...", "confidence": ..., "raw_score": ...}`
- `GET /health` — a simple health check endpoint

```bash
python3 src/app.py
# then open http://localhost:8080
```

Every prediction is logged as structured output — the "prediction telemetry" pillar of observability, the foundation for drift detection in a full production setup.

---

## Containerization (Docker)

The book's Docker best practices, applied directly:

- **Slim base image** (`python:3.11-slim`) — smaller size, less attack surface
- **Dependencies installed before code is copied** — so Docker's layer cache is reused when only code changes, speeding up rebuilds
- **Non-root user** — the container runs as a limited user, not root, for defense in depth

```bash
docker build -t pneumonia-detector:1.0 .
docker run -p 8080:8080 pneumonia-detector:1.0
```

The same demo, running identically inside the container as it does locally — proof the environment is fully reproducible.

---

## Planned Cloud Deployment (AWS)

The book maps a short, consistent chain of AWS services for a pipeline like this one:

| Service | Role |
|---|---|
| **S3** | Stores raw data, artifacts, trained model files |
| **ECR** | Stores the Docker image |
| **Lambda** | Runs the container as a serverless, callable service |
| **CloudWatch** | Collects logs and prediction metrics |

Deployment commands (not yet executed in this project, given time constraints — included here as the exact planned path):

```bash
aws ecr create-repository --repository-name pneumonia-detector --region us-east-1
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag pneumonia-detector:1.0 <account-id>.dkr.ecr.us-east-1.amazonaws.com/pneumonia-detector:1.0
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/pneumonia-detector:1.0
aws lambda create-function --function-name pneumonia-predict --package-type Image \
  --code ImageUri=<account-id>.dkr.ecr.us-east-1.amazonaws.com/pneumonia-detector:1.0 \
  --role arn:aws:iam::<account-id>:role/pneumonia-lambda-role --timeout 30 --memory-size 1024
aws lambda create-function-url-config --function-name pneumonia-predict --auth-type NONE
```

---

## CI/CD

The book's CI lesson: keep it fast — lint, then unit test, then a build check, never a full retrain on every push. `.github/workflows/ci.yml` runs this automatically on every push to `main`:

1. **Lint** (`flake8`) — style and syntax checks
2. **Unit tests** (`tests/test_data.py`) — fast tests on the core preprocessing logic, using small generated test images, not the full dataset
3. **Docker build check** — confirms the Dockerfile still builds successfully

On the first real run, CI caught genuine style issues in the codebase; they were fixed push by push until every check passed — CI doing its actual job of catching mistakes before they reach anything real.

```bash
cd tests && python3 test_data.py   # run tests locally
```

---

## Project Structure

```
pneumonia-detector/
├── data/
│   └── chest_xray/            # dataset (train/test, NORMAL/PNEUMONIA)
├── src/
│   ├── validate_data.py       # data validation
│   ├── train.py                # model training
│   ├── app.py                  # Flask API + frontend route
│   └── templates/
│       └── index.html          # demo frontend
├── tests/
│   └── test_data.py            # CI unit tests
├── .github/workflows/
│   └── ci.yml                   # CI pipeline definition
├── Dockerfile
├── requirements.txt
├── pneumonia_model.h5           # trained model (generated by train.py)
└── README.md
```

---

## Setup & Running Locally

```bash
# 1. Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Validate the data
python3 src/validate_data.py

# 4. Train the model
python3 src/train.py

# 5. Run the app
python3 src/app.py
# open http://localhost:8080
```

---

## What's Not Yet Done, and Why

- **SHAP/LIME interpretability** — attempted, but blocked by a known `llvmlite` build failure on macOS. Deprioritized given the project deadline rather than left silently incomplete.
- **Live AWS deployment** — the exact commands are documented above and the Dockerfile is deployment-ready, but the ECR/Lambda steps were not executed in the time available.

---

## Disclaimer

This is a student project demonstrating an MLOps pipeline, not a certified diagnostic device. Predictions should never replace a radiologist's assessment.
