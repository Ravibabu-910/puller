# medical-chatbot

## Table Image -> JSON Extractor (CPU, local)

This repo now includes a **local CPU pipeline** that combines:
- **OpenCV** for table structure detection.
- **Tesseract OCR** for text extraction.
- **Local LLM (Ollama)** for schema-normalized JSON output.
- **FastAPI** endpoints for extraction and expected-vs-predicted comparison.

> Goal: extract clinical/any tabular image into JSON close to your target format.

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install system OCR dependency:

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

Install/start Ollama and pull a local model (example):

```bash
ollama serve
ollama pull llama3.1:8b
```

## 2) Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## 3) Extract JSON from image

```bash
curl -X POST http://localhost:8000/extract \
  -F "image=@/path/to/table.png" \
  -F "schema_hint=[{\"assessment\":\"\",\"outcome_type\":\"\",\"PBO_n96\":\"\",\"ABBV154_40mg_Q2W_n98\":\"\",\"ABBV154_150mg_Q2W_n94\":\"\",\"ABBV154_340mg_Q2W_n90\":\"\"}]" \
  -F "llm_model=llama3.1:8b" \
  -F "ollama_url=http://localhost:11434"
```

Response includes:
- `raw_grid`: OCR-reconstructed table matrix.
- `raw_rows`: row objects before final standardization.
- `json_output`: LLM-standardized JSON in your requested format.
- `notes`: warnings/fallback info.

## 4) Compare extracted vs expected JSON

```bash
curl -X POST http://localhost:8000/compare \
  -F 'predicted={"a":"1"}' \
  -F 'expected={"a":"1"}'
```

Returns field-level fuzzy score (%), match counts, and mismatch details.

## 5) CLI mode

```bash
python scripts/run_extract.py /path/to/table.png \
  --schema '[{"assessment":"","outcome_type":""}]' \
  --model llama3.1:8b \
  --ollama-url http://localhost:11434
```

## Accuracy notes

- For high accuracy (target >90%), use:
  - high-resolution image (>=200 DPI),
  - clean crop around table,
  - explicit `schema_hint`,
  - stronger local model if available (e.g., `llama3.1:70b` if hardware supports).
- If LLM call fails, code falls back to a heuristic JSON builder so extraction still returns structured output.

