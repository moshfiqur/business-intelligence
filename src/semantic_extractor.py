"""Column semantics extractor.

Reads a CSV file, infers per-column metadata, and writes the semantics to
``output/column_semantics/<dataset>.csv`` for downstream consumption.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnDescriptor:
    """Structured view of a single column's semantics."""

    column_name: str
    data_type: str
    semantic_category: str
    sample_values: List[str]

    def to_record(self) -> dict:
        return {
            "column_name": self.column_name,
            "data_type": self.data_type,
            "semantic_category": self.semantic_category,
            "sample_values": json.dumps(self.sample_values, ensure_ascii=False),
        }


class ColumnSemanticsExtractor:
    """Extracts column-level semantics from tabular data."""

    def __init__(self, output_dir: str = "output/column_semantics", sample_size: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size

    def extract(self, dataframe: pd.DataFrame) -> List[ColumnDescriptor]:
        descriptors: List[ColumnDescriptor] = []
        for column in dataframe.columns:
            series = dataframe[column]
            descriptor = ColumnDescriptor(
                column_name=column,
                data_type=str(series.dtype),
                semantic_category=self._infer_semantic_category(column, series),
                sample_values=self._get_sample_values(series),
            )
            descriptors.append(descriptor)
        return descriptors

    def process_file(self, csv_path: str, dataset_name: str | None = None) -> Path:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        dataset = self._derive_dataset_name(csv_file, dataset_name)
        logger.info("Loading dataset '%s' from %s", dataset, csv_file)
        dataframe = pd.read_csv(csv_file)

        descriptors = self.extract(dataframe)
        semantics_df = pd.DataFrame([descriptor.to_record() for descriptor in descriptors])

        output_path = self.output_dir / f"{dataset}.csv"
        semantics_df.to_csv(output_path, index=False)
        logger.info("Wrote column semantics for '%s' to %s", dataset, output_path)
        return output_path

    def _derive_dataset_name(self, csv_file: Path, override: str | None) -> str:
        if override:
            return self._safe_name(override)
        return self._safe_name(csv_file.stem)

    def _safe_name(self, value: str) -> str:
        sanitized = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip()]
        name = "".join(sanitized).strip("_")
        return name or "dataset"

    def _get_sample_values(self, series: pd.Series) -> List[str]:
        clean_series = series.dropna()
        if clean_series.empty:
            return []

        unique_values = clean_series.unique().tolist()
        if len(unique_values) <= self.sample_size:
            return [str(value) for value in unique_values]

        frequent = clean_series.value_counts().index.tolist()[: max(1, self.sample_size // 2)]
        remaining_needed = self.sample_size - len(frequent)
        remainder_pool = clean_series[~clean_series.isin(frequent)]
        if remainder_pool.empty:
            random_samples: List[str] = []
        else:
            random_samples = remainder_pool.sample(
                n=remaining_needed,
                replace=len(remainder_pool) < remaining_needed,
                random_state=42,
            ).tolist()

        samples = (frequent + random_samples)[: self.sample_size]
        return [str(sample) for sample in samples]

    def _infer_semantic_category(self, column_name: str, series: pd.Series) -> str:
        column_lower = column_name.lower()
        categories = {
            "financial": ["price", "cost", "revenue", "amount", "salary", "budget"],
            "temporal": ["date", "time", "year", "month", "day", "timestamp"],
            "identifier": ["id", "code", "number", "key", "reference"],
            "person": ["name", "employee", "customer", "user", "person"],
            "location": ["address", "city", "country", "location", "region"],
            "product": ["product", "item", "sku", "inventory", "stock"],
            "status": ["status", "flag", "active", "inactive", "approved"],
        }

        for category, keywords in categories.items():
            if any(keyword in column_lower for keyword in keywords):
                return category

        non_null = series.dropna()
        if non_null.empty:
            return "unknown"
        if pd.api.types.is_datetime64_any_dtype(non_null):
            return "temporal"
        if pd.api.types.is_numeric_dtype(non_null):
            return "numeric"
        cardinality_ratio = len(non_null.unique()) / max(len(non_null), 1)
        if cardinality_ratio <= 0.1:
            return "categorical"
        return "textual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract column semantics from a CSV file.")
    parser.add_argument("csv_path", help="Path to the CSV file to analyze.")
    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        help="Optional dataset name to use for the output file.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/column_semantics",
        help="Directory where the column semantics CSV should be stored.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of sample values to store for each column.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    extractor = ColumnSemanticsExtractor(
        output_dir=args.output_dir,
        sample_size=args.sample_size,
    )
    output_path = extractor.process_file(args.csv_path, args.dataset_name)
    print(f"Column semantics saved to {output_path}")


if __name__ == "__main__":
    main()
