from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractResponse(BaseModel):
    raw_grid: list[list[str]] = Field(default_factory=list)
    raw_rows: list[dict[str, Any]] = Field(default_factory=list)
    json_output: Any
    notes: list[str] = Field(default_factory=list)


class CompareRequest(BaseModel):
    predicted: Any
    expected: Any


class CompareResponse(BaseModel):
    score: float
    field_matches: int
    field_total: int
    details: list[str] = Field(default_factory=list)
