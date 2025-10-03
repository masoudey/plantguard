# PlantGuard Multimodal Chatbot

PlantGuard is a proof-of-concept multimodal assistant that helps smallholder farmers diagnose foliar plant diseases using image, speech, and text inputs. The project demonstrates the full AI lifecycle required in the assignment brief, from data preparation and model training to deployment, ethics, and reporting.

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
   - Audio: place WAV files under `data/processed/audio/<split>/...` and create `train.json` / `val.json` manifests describing file paths and labels.
   - FAQ: add SQuAD-style `train.json` (and optional `validation.json`) to `data/processed/faq/`.
   - Knowledge base: store agronomy notes in `data/knowledge_base/` (`.txt`, `.md`, or JSON Q&A pairs`).
6. **Train core models**
   - Vision: `python scripts/train_vision.py --data-dir data/processed/plantvillage --epochs 15 --batch-size 32`
   - Audio:  `python scripts/train_audio.py --data-dir data/processed/audio --epochs 20 --batch-size 32`
   - Text (fast dev run): `python scripts/train_text.py --data-dir data/processed/faq --model distilbert-base-uncased --epochs 1 --batch-size 8 --max-length 256 --doc-stride 64 --sample-size 2000`
7. **Build the LangChain FAISS knowledge base**  
   `python scripts/build_vector_store.py --input data/knowledge_base --output models/text/knowledge_base`
8. **Optional: train multimodal fusion** once you have aligned feature packs  
   `python scripts/train_fusion.py --train data/processed/fusion/train.pt --val data/processed/fusion/val.pt`
9. **Run the application**
   - Backend only: `uvicorn plantguard.src.backend.main:app --reload`
   - Frontend dev server: `cd frontend && npm install && npm run dev`
   - Docker (backend + frontend): `docker compose -f docker/docker-compose.yml up --build`

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
  1. Create manifest files `data/processed/audio/train.json` (and optionally `val.json`) containing records such as `{"file": "train/clip_001.wav", "label": "powdery_mildew"}`. Paths are resolved relative to `data/processed/audio`.
  2. Launch training:
     ```bash
     python scripts/train_audio.py --data-dir data/processed/audio --epochs 20 --batch-size 32
     ```
     The checkpoint `models/audio/plantguard_cnn_lstm.pt` stores learned weights, label order, and MFCC padding length used at inference time.

- **Text QA (BERT fine-tuning)**
  1. Supply `data/processed/faq/train.json` (and optionally `validation.json`) in SQuAD-style format.
  2. Fine-tune the head with:
     ```bash
     python scripts/train_text.py --data-dir data/processed/faq --epochs 3 --batch-size 8
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

## Assignment Mapping
- **Environment Setup**: `pyproject.toml`, Docker assets, Makefile, and `frontend/package.json` document the environment configuration.
- **Data Exploration**: See notebooks in `notebooks/` and dataset utilities in `scripts/download_data.py`.
- **Preprocessing Pipelines**: Located under `src/backend/services/preprocess.py` with modality-specific helpers.
- **Model Design**: Vision, audio, text, and fusion models live under `src/backend/services/`.
- **Training & Evaluation**: Scripts in `scripts/` handle modality training, fusion, and evaluation metrics.
- **Deployment & UI**: FastAPI backend and React SPA demonstrate a multimodal workflow; Docker assets support containerisation.
- **Ethics & Compliance**: Guidance and placeholders included in `reports/` and config files to reference privacy/bias mitigations.

## Reporting
The `reports/` directory contains a BSBI template placeholder and space for figures. Export final reports to PDF/DOCX per submission guidelines.
