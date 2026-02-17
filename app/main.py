from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile

from app.schemas import CompareResponse, ExtractResponse
from app.table_pipeline import (
    cells_to_grid,
    compare_json,
    detect_cells,
    grid_to_rows,
    llm_standardize,
    preprocess_image,
)

app = FastAPI(title="Table OCR + LLM Extractor", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
async def extract_table(
    image: UploadFile = File(...),
    schema_hint: str | None = Form(default=None),
    llm_model: str = Form(default="llama3.1:8b"),
    ollama_url: str = Form(default="http://localhost:11434"),
) -> ExtractResponse:
    image_bytes = await image.read()

    binary = preprocess_image(image_bytes)
    cells = detect_cells(binary)
    grid = cells_to_grid(cells)
    rows = grid_to_rows(grid)

    structured, warning = llm_standardize(rows, schema_hint, llm_model, ollama_url)
    notes = [warning] if warning else []

    return ExtractResponse(raw_grid=grid, raw_rows=rows, json_output=structured, notes=notes)


@app.post("/compare", response_model=CompareResponse)
def compare(predicted: str = Form(...), expected: str = Form(...)) -> CompareResponse:
    pred_obj: Any = json.loads(predicted)
    exp_obj: Any = json.loads(expected)
    score, matches, total, details = compare_json(pred_obj, exp_obj)
    return CompareResponse(score=score, field_matches=matches, field_total=total, details=details)
