from __future__ import annotations

import argparse
import json

from app.table_pipeline import (
    cells_to_grid,
    detect_cells,
    grid_to_rows,
    llm_standardize,
    preprocess_image,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured JSON from a table image")
    parser.add_argument("image_path")
    parser.add_argument("--schema", default="", help="Schema hint JSON or text")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    with open(args.image_path, "rb") as f:
        image_bytes = f.read()

    bw = preprocess_image(image_bytes)
    cells = detect_cells(bw)
    grid = cells_to_grid(cells)
    rows = grid_to_rows(grid)
    result, warning = llm_standardize(rows, args.schema, args.model, args.ollama_url)

    output = {
        "raw_rows": rows,
        "json_output": result,
        "warning": warning,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
