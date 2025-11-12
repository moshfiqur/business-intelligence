"""Zero-shot column semantics extractor.

Reads a CSV file, infers per-column metadata with a zero-shot classifier, and
writes the semantics to ``output/column_semantics/<dataset>.csv`` for downstream
consumption.

Sample command:
python src/semantic_extractor.py data/pavansubhasht/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv \
    --dataset-name ibm-hr-analytics-attrition-dataset \
    --output-dir output/column_semantics \
    --sample-size 5 \
    --model-name valhalla/distilbart-mnli-12-1 \
    --confidence-threshold 0.5 \
    --log-level INFO
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from transformers import pipeline

logger = logging.getLogger(__name__)


DEFAULT_LABEL_PROMPTS: Dict[str, str] = {
    "financial": "financial metrics such as revenue, cost, profit, or budgets",
    "compensation": "employee compensation including salary bands, bonuses, or stock options",
    "temporal": "temporal attributes like dates, months, timestamps, or durations",
    "identifier": "identifier fields such as ids, keys, or reference numbers",
    "person": "people-related attributes such as employee names, customer demographics, or contacts",
    "location": "geographic information like city, country, region, or address",
    "product": "product or inventory descriptors such as sku, item, or catalog details",
    "status": "status or lifecycle indicators such as active, closed, approved, or flags",
    "performance": "performance indicators like kpis, targets, or scores",
    "operations": "operational metrics such as quantity, fulfillment, logistics, or capacity",
}


@dataclass
class ColumnDescriptor:
    """Structured view of a single column's semantics."""

    column_name: str
    data_type: str
    normalized_dtype: str
    semantic_category: str
    semantic_confidence: float | None
    sample_values: List[str]

    def to_record(self) -> dict:
        return {
            "column_name": self.column_name,
            "data_type": self.data_type,
            "normalized_dtype": self.normalized_dtype,
            "semantic_category": self.semantic_category,
            "semantic_confidence": (
                round(self.semantic_confidence, 4) if self.semantic_confidence is not None else ""
            ),
            "sample_values": json.dumps(self.sample_values, ensure_ascii=False),
        }


class ColumnSemanticsExtractor:
    """Extracts column-level semantics from tabular data."""

    def __init__(
        self,
        output_dir: str = "output/column_semantics",
        sample_size: int = 5,
        model_name: str = "valhalla/distilbart-mnli-12-1",
        label_config: str | None = None,
        hypothesis_template: str = "This column describes {}.",
        confidence_threshold: float = 0.45,
        device: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.hypothesis_template = hypothesis_template
        self.confidence_threshold = confidence_threshold
        self.label_prompts = self._load_label_prompts(label_config)
        if not self.label_prompts:
            raise ValueError("At least one semantic label must be configured")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=self._resolve_device(device),
        )

    def extract(self, dataframe: pd.DataFrame) -> List[ColumnDescriptor]:
        descriptors: List[ColumnDescriptor] = []
        for column in dataframe.columns:
            series = dataframe[column]
            sample_values = self._get_sample_values(series)
            semantic_category, confidence = self._classify_semantic_category(
                column_name=column,
                series=series,
                sample_values=sample_values,
            )
            descriptor = ColumnDescriptor(
                column_name=column,
                data_type=str(series.dtype),
                normalized_dtype=self._normalize_dtype(series),
                semantic_category=semantic_category,
                semantic_confidence=confidence,
                sample_values=sample_values,
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

    def _normalize_dtype(self, series: pd.Series) -> str:
        dtype = series.dtype
        if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            return "string"
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        if pd.api.types.is_float_dtype(dtype):
            return "float"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        if pd.api.types.is_categorical_dtype(dtype):  # pragma: no cover - optional
            return "categorical"
        if pd.api.types.is_timedelta64_dtype(dtype):
            return "timedelta"
        return str(dtype)

    def _classify_semantic_category(
        self,
        column_name: str,
        series: pd.Series,
        sample_values: List[str],
    ) -> Tuple[str, float | None]:
        sequence = column_name.replace("_", " ").strip()
        if sample_values:
            preview = ", ".join(sample_values[:3])
            sequence = f"{sequence}. Sample values: {preview}"

        candidate_labels, label_lookup = self._build_candidate_labels()
        if not candidate_labels:
            return self._heuristic_category(column_name, series), None

        try:
            result = self.classifier(
                sequences=sequence,
                candidate_labels=candidate_labels,
                multi_label=True,
                hypothesis_template=self.hypothesis_template,
            )
            best_label_key = result["labels"][0]
            best_score = float(result["scores"][0])
            mapped_label = label_lookup.get(best_label_key, best_label_key)
            if best_score >= self.confidence_threshold:
                return mapped_label, best_score
            logger.debug(
                "Low confidence %.3f for column '%s'; falling back to heuristics",
                best_score,
                column_name,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Zero-shot classification failed for '%s': %s", column_name, exc)

        return self._heuristic_category(column_name, series), None

    def _build_candidate_labels(self) -> Tuple[List[str], Dict[str, str]]:
        candidate_labels: List[str] = []
        label_lookup: Dict[str, str] = {}
        for label, description in self.label_prompts.items():
            label_text = description.strip() if description else label
            if not label_text:
                continue
            candidate_labels.append(label_text)
            label_lookup[label_text] = label
        return candidate_labels, label_lookup

    def _heuristic_category(self, column_name: str, series: pd.Series) -> str:
        column_lower = column_name.lower()
        categories = {
            "financial": ["price", "cost", "revenue", "amount", "budget"],
            "compensation": ["salary", "bonus", "stock option", "equity", "compensation"],
            "temporal": ["date", "time", "year", "month", "day", "timestamp"],
            "identifier": ["id", "code", "number", "key", "reference"],
            "person": ["name", "employee", "customer", "user", "person"],
            "location": ["address", "city", "country", "location", "region"],
            "product": ["product", "item", "sku", "inventory"],
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

    def _load_label_prompts(self, label_config: str | None) -> Dict[str, str]:
        prompts = dict(DEFAULT_LABEL_PROMPTS)
        if not label_config:
            return prompts

        config_path = Path(label_config)
        if not config_path.exists():
            raise FileNotFoundError(f"Label config not found: {config_path}")

        data = self._read_label_file(config_path)
        if not isinstance(data, dict):
            raise ValueError("Label config must define a mapping of label -> prompt")
        for label, description in data.items():
            prompts[str(label)] = str(description)
        return prompts

    def _read_label_file(self, path: Path):
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "PyYAML is required to load YAML label configs. Install it via 'pip install pyyaml'."
                ) from exc
            return yaml.safe_load(text)
        return json.loads(text)

    def _resolve_device(self, device_str: str | None) -> int:
        if device_str is None:
            return -1  # CPU by default for wider compatibility
        normalized = device_str.strip().lower()
        if normalized == "cpu":
            return -1
        if normalized in {"cuda", "gpu"}:
            return 0
        try:
            return int(normalized)
        except ValueError:
            logger.warning("Unrecognized device '%s'; defaulting to CPU", device_str)
            return -1


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
        "--model-name",
        default="valhalla/distilbart-mnli-12-1",
        help="Zero-shot model to use for semantic classification.",
    )
    parser.add_argument(
        "--label-config",
        help="Optional JSON/YAML file mapping label->prompt to override defaults.",
    )
    parser.add_argument(
        "--hypothesis-template",
        default="This column describes {}.",
        help="Template passed to the zero-shot classifier.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.45,
        help="Minimum model confidence required before falling back to heuristics.",
    )
    parser.add_argument(
        "--device",
        help="Force a specific device id (e.g. 'cpu', 'cuda', '0'). Defaults to CPU.",
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
        model_name=args.model_name,
        label_config=args.label_config,
        hypothesis_template=args.hypothesis_template,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )
    output_path = extractor.process_file(args.csv_path, args.dataset_name)
    print(f"Column semantics saved to {output_path}")


if __name__ == "__main__":
    main()
