# PlantGuard Multimodal Chatbot

PlantGuard is a proof-of-concept multimodal assistant that helps smallholder farmers diagnose foliar plant diseases using image, speech, and text inputs. The project demonstrates the full AI lifecycle from data preparation and model training to deployment, ethics, and reporting.

## Repository Layout
```
plantguard/
├── docker/                  # Containerisation assets
├── frontend/                # React + Vite + Tailwind CSS single-page application
├── src/                     # Python source code (FastAPI backend and services)
├── scripts/                 # Training and utility scripts
├── notebooks/               # Research and exploration notebooks
├── data/                    # Raw and processed datasets (gitignored)
├── models/                  # Trained weights and checkpoints (gitignored)
├── tests/                   # Automated Python tests
├── reports/                 # Assignment report assets
├── logs/                    # Runtime logs (gitignored)
└── linting/                 # Shared linting configuration
```

## Quick Start
1. **Clone & enter the project**  
   `git clone https://github.com/masoudey/plantguard.git`  
   `cd plantguard`
2. **Create a virtual environment (Python 3.10+)**  
   `python -m venv .venv`  
   `source .venv/bin/activate`
3. **Install dependencies**  
   `pip install -r requirements.txt`  
   `pip install langchain-huggingface sentence-transformers accelerate`
4. **Copy environment defaults**  
   `cp .env.example .env` and fill in any optional API keys.
5. **Prepare datasets**
   - Vision: extract PlantVillage into `data/raw/plantvillage/extracted/` and run `python scripts/prepare_plantvillage.py --raw-dir data/raw/plantvillage/extracted --output data/processed/plantvillage`.
   - Audio: follow the [Audio Dataset Generation](#audio-dataset-generation) workflow to script symptom sentences, synthesize WAV clips, and emit `train.json` / `val.json` manifests.
   - FAQ: add SQuAD-style `train.json` (and optional `validation.json`) to `data/processed/faq/`.
   - Knowledge base: store agronomy notes in `data/knowledge_base/` (`.txt`, `.md`, or JSON Q&A pairs`).
6. **Train core models**
   - Vision: `python scripts/train_vision.py --data-dir data/processed/plantvillage --epochs 15 --batch-size 32`
   - Audio:
     ```bash
     python scripts/train_audio.py \
       --data-dir data/processed/audio \
       --epochs 20 \
       --batch-size 32 \
       --num-workers 0 \
       --log-dir runs/audio \
       --metrics-output reports/audio_best.json
     ```
  - Text (fast dev run):
    ```bash
    .venv/bin/python scripts/train_text.py \
      --data-dir data/processed/faq \
      --model distilbert-base-uncased \
      --epochs 1 \
      --batch-size 8 \
      --max-length 256 \
      --doc-stride 64 \
      --sample-size 2000
    ```
7. **Build the LangChain FAISS knowledge base**  
   `python scripts/build_vector_store.py --input data/knowledge_base --output models/text/knowledge_base`
8. **Optional: train multimodal fusion** once you have aligned feature packs  
   `python scripts/train_fusion.py --train data/processed/fusion/train.pt --val data/processed/fusion/val.pt`
9. **Run the application**
 - Backend only: `uvicorn plantguard.src.backend.main:app --reload`
  - Frontend dev server: `cd frontend && npm install && npm run dev`
  - Docker (backend + frontend): `docker compose -f docker/docker-compose.yml up --build`

## Audio Dataset Generation
- **Draft symptom narratives**: Populate `plantguard/data/audio/Expanded_Plant_Symptom_TTS_Manifest.csv` with concise field observations (`text`), target class labels (`label`), and explicit output paths such as `train/powdery_mildew/powdery_mildew_001.wav` or `val/powdery_mildew/powdery_mildew_val_001.wav`. Include per-row voice/language overrides if you want accented variants.
- **Synthesize speech clips**:
  ```bash
  python plantguard/scripts/generate_audio_from_csv.py \
    --csv plantguard/data/audio/Expanded_Plant_Symptom_TTS_Manifest.csv \
    --output-dir plantguard/data/processed/audio \
    --mono
  ```
  The helper writes 16 kHz mono WAVs beneath `data/processed/audio/<split>/<label>/` and mirrors the rel paths defined in the CSV.
- **Regenerate manifests**: After audio export, rebuild manifests so the trainer has balanced metadata.
  ```bash
  python - <<'PY'
from pathlib import Path
import json

root = Path('data/processed/audio')
for split in ['train', 'val']:
    base = root / split
    if not base.exists():
        continue
    records = [
        {"file": str(path.relative_to(root)), "label": path.parent.name}
        for path in base.rglob('*.wav')
    ]
    (root / f'{split}.json').write_text(json.dumps(records, indent=2))
    print(f"Wrote {split}.json with {len(records)} items")
PY
  ```
- **Iterate for coverage**: Repeat the draft → synthesize → manifest loop until each label has ample train/val clips. Mix phrasing, speech rates, and voices to improve generalisation.

## Model Training
Each modality ships with a training script under `scripts/` and expects pre-processed data in `data/processed/`.

- **Vision (ResNet50 fine-tuning)**
  1. Arrange PlantVillage-style images under `data/processed/plantvillage/<split>/<class>/<image>.jpg`. If you only have a single folder, the script will carve out a validation split automatically.
     ```bash
     # Example workflow after downloading PlantVillage (zip) locally
     unzip PlantVillage-Dataset.zip -d data/raw/plantvillage/extracted
     python scripts/prepare_plantvillage.py --raw-dir data/raw/plantvillage/extracted --output data/processed/plantvillage
     ```
  2. Train and save weights with:
     ```bash
     python scripts/train_vision.py --data-dir data/processed/plantvillage --epochs 15 --batch-size 32
     ```
     The script writes `models/vision/plantguard_resnet50.pt`, embedding the class names for inference.

- **Audio (CNN-LSTM on MFCCs)**
  1. Confirm `data/processed/audio/train/<label>/` and `val/<label>/` mirror the rel paths defined in the CSV/manifests generated earlier.
  2. Launch training (set `--num-workers 0` on macOS/MPS hosts to avoid multiprocessing import issues):
     ```bash
     python scripts/train_audio.py \
       --data-dir data/processed/audio \
       --epochs 20 \
       --batch-size 32 \
       --num-workers 0 \
       --log-dir runs/audio \
       --metrics-output reports/audio_best.json
     ```
     The trainer logs loss/accuracy/macro-F1 (TensorBoard when available), persists per-class precision/recall curves in `reports/audio_best.json`, and saves the best checkpoint with label metadata to `models/audio/plantguard_cnn_lstm.pt`.

- **Text QA (BERT fine-tuning)**
  1. Supply `data/processed/faq/train.json` (and optionally `validation.json`) in SQuAD-style format.
  2. Fine-tune the head with:
     ```bash
     .venv/bin/python scripts/train_text.py \
       --data-dir data/processed/faq \
       --model bert-base-uncased \
       --epochs 3 \
       --batch-size 16 \
       --max-length 384 \
       --doc-stride 128 \
       --sample-size 20000
     ```
     Outputs land in `models/text/plantguard_qa_head/` alongside the tokenizer.

- **Knowledge Base (LangChain + FAISS RAG)**
  1. Place curated agronomy documents under `data/knowledge_base/` (`.txt`, `.md`, or JSON lists of QA pairs).
  2. Build the FAISS index and metadata bundle with:
     ```bash
     python scripts/build_vector_store.py --input data/knowledge_base --output models/text/knowledge_base
     ```
     The script relies on LangChain text splitters, Hugging Face embeddings, and the FAISS vector store to persist documents under `models/text/knowledge_base/`. The backend loads this directory automatically and provides grounded answers with citations when available.

After training the individual modalities you can experiment with multimodal fusion by generating aligned feature triples. The fusion trainer expects feature packs created via `torch.save` with keys `image`, `text`, `audio` (optional), `labels`, and `class_names`. Launch training with, for example:

```bash
python scripts/train_fusion.py --train data/processed/fusion/train.pt --val data/processed/fusion/val.pt
```

Rebuild the containers (`docker compose -f docker/docker-compose.yml up --build`) so the API loads the new checkpoints.

## Docker Workflow
- `docker/docker-compose.yml` builds and serves both services. Launch with `docker compose -f docker/docker-compose.yml up --build`.
- The backend container runs `uvicorn` on `http://localhost:8000`. The frontend container serves the built React/Tailwind app via Nginx on `http://localhost:3000`.
- During development you can still run `npm run dev` locally for hot reloading if desired.

##  Mapping
- **Environment Setup**: `pyproject.toml`, Docker assets, Makefile, and `frontend/package.json` document the environment configuration.
- **Data Exploration**: See notebooks in `notebooks/` and dataset utilities in `scripts/download_data.py`.
- **Preprocessing Pipelines**: Located under `src/backend/services/preprocess.py` with modality-specific helpers.
- **Model Design**: Vision, audio, text, and fusion models live under `src/backend/services/`.
- **Training & Evaluation**: Scripts in `scripts/` handle modality training, fusion, and evaluation metrics.
- **Deployment & UI**: FastAPI backend and React SPA demonstrate a multimodal workflow; Docker assets support containerisation.
- **Ethics & Compliance**: Guidance and placeholders included in `reports/` and config files to reference privacy/bias mitigations.

## Reporting
The `reports/` directory contains a BSBI template placeholder and space for figures. Export final reports to PDF/DOCX per submission guidelines.
