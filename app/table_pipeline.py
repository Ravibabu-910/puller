from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import cv2
import httpx
import numpy as np
import pytesseract
from rapidfuzz import fuzz


@dataclass
class Cell:
    x: int
    y: int
    w: int
    h: int
    text: str


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("|", "")
    return text


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    return bw


def detect_cells(binary_img: np.ndarray) -> list[Cell]:
    h, w = binary_img.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 25), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 25)))

    horizontal = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    grid = cv2.add(horizontal, vertical)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells: list[Cell] = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 35 or ch < 18:
            continue
        if cw > w * 0.98 and ch > h * 0.98:
            continue

        roi = 255 - binary_img[y : y + ch, x : x + cw]
        text = pytesseract.image_to_string(roi, config="--oem 3 --psm 6")
        text = _clean_text(text)
        if not text:
            continue
        cells.append(Cell(x=x, y=y, w=cw, h=ch, text=text))

    if not cells:
        text = pytesseract.image_to_string(255 - binary_img, config="--oem 3 --psm 6")
        lines = [_clean_text(line) for line in text.splitlines() if _clean_text(line)]
        return [Cell(0, i * 20, 100, 20, t) for i, t in enumerate(lines)]

    return cells


def cells_to_grid(cells: list[Cell], y_tol: int = 12, x_tol: int = 15) -> list[list[str]]:
    if not cells:
        return []

    row_buckets: list[list[Cell]] = []
    for cell in sorted(cells, key=lambda c: c.y):
        placed = False
        for bucket in row_buckets:
            if abs(bucket[0].y - cell.y) <= y_tol:
                bucket.append(cell)
                placed = True
                break
        if not placed:
            row_buckets.append([cell])

    x_centers = sorted([c.x for c in cells])
    col_positions: list[int] = []
    for x in x_centers:
        if not col_positions or abs(col_positions[-1] - x) > x_tol:
            col_positions.append(x)

    grid: list[list[str]] = []
    for row_cells in row_buckets:
        row = ["" for _ in col_positions]
        for cell in sorted(row_cells, key=lambda c: c.x):
            idx = min(range(len(col_positions)), key=lambda i: abs(col_positions[i] - cell.x))
            if row[idx]:
                row[idx] = f"{row[idx]} {cell.text}".strip()
            else:
                row[idx] = cell.text
        if any(v.strip() for v in row):
            grid.append([_clean_text(v) for v in row])
    return grid


def grid_to_rows(grid: list[list[str]]) -> list[dict[str, Any]]:
    if not grid:
        return []

    header_idx = 0
    for i, row in enumerate(grid[:6]):
        if sum(1 for c in row if c) >= 3:
            header_idx = i
            break

    header = [c if c else f"col_{i}" for i, c in enumerate(grid[header_idx])]
    rows = []
    for row in grid[header_idx + 1 :]:
        if not any(row):
            continue
        rows.append({header[i]: row[i] if i < len(row) else "" for i in range(len(header))})
    return rows


def _heuristic_json(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current_outcome = ""
    for row in rows:
        values = [v for v in row.values() if isinstance(v, str)]
        row_text = " | ".join(values)

        if "proportion" in row_text.lower() and "week" in row_text.lower():
            current_outcome = row_text
            continue

        numeric_cols = [v for v in values if re.search(r"\d", v)]
        if len(numeric_cols) < 2:
            continue

        assessment = values[0]
        record: dict[str, str] = {
            "assessment": assessment,
            "outcome_type": current_outcome or "table_outcome",
        }

        arm_idx = 0
        for key, val in row.items():
            if not re.search(r"\d", str(val)):
                continue
            col_key = key or f"arm_{arm_idx}"
            norm_key = re.sub(r"[^A-Za-z0-9]+", "_", col_key).strip("_")
            record[norm_key] = str(val)
            arm_idx += 1

        records.append(record)
    return records


def llm_standardize(
    rows: list[dict[str, Any]],
    schema_hint: str | None,
    model: str,
    ollama_url: str,
) -> tuple[Any, str | None]:
    prompt = (
        "You are an information extraction engine for medical trial tables. "
        "Convert the provided parsed table rows to clean JSON. "
        "If schema hint is provided, strictly match it. "
        "Keep values exactly as in source with confidence intervals/significance. "
        "Return JSON only."
    )

    user_content = {
        "schema_hint": schema_hint or "No explicit schema. Return an array of objects.",
        "rows": rows,
    }

    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(
                f"{ollama_url.rstrip('/')}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(user_content)},
                    ],
                    "stream": False,
                    "format": "json",
                },
            )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        parsed = json.loads(content)
        return parsed, None
    except Exception as exc:  # noqa: BLE001
        fallback = _heuristic_json(rows)
        return fallback, f"LLM standardization failed, fallback used: {exc}"


def flatten_json(data: Any, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, new_key))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{prefix}[{i}]"
            out.update(flatten_json(v, new_key))
    else:
        out[prefix] = str(data)
    return out


def compare_json(predicted: Any, expected: Any) -> tuple[float, int, int, list[str]]:
    p = flatten_json(predicted)
    e = flatten_json(expected)

    matches = 0
    details: list[str] = []
    for k, exp_val in e.items():
        pred_val = p.get(k)
        if pred_val is None:
            details.append(f"Missing key: {k}")
            continue
        similarity = fuzz.ratio(pred_val, exp_val)
        if similarity >= 90:
            matches += 1
        else:
            details.append(f"Mismatch {k}: expected='{exp_val}' predicted='{pred_val}' ({similarity:.1f})")

    total = max(len(e), 1)
    score = (matches / total) * 100.0
    return round(score, 2), matches, total, details
