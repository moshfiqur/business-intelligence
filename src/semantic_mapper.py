"""Semantic mapper.

Reads column semantics CSV files, computes cross-dataset column similarity maps,
and stores the results in ``output/semantic_mappings``.
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ColumnSimilarityMapper:
    """Build semantic mappings across datasets using column metadata."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dir: str = "output/semantic_mappings",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_semantics(self, semantics_dir: str) -> Dict[str, pd.DataFrame]:
        directory = Path(semantics_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Semantics directory not found: {directory}")

        semantics: Dict[str, pd.DataFrame] = {}
        for csv_file in sorted(directory.glob("*.csv")):
            dataset_name = csv_file.stem
            dataframe = pd.read_csv(csv_file)
            if "column_name" not in dataframe.columns:
                logger.warning("Skipping %s because it lacks a 'column_name' column", csv_file)
                continue
            if "semantic_category" not in dataframe.columns:
                logger.warning(
                    "Skipping %s because it lacks a 'semantic_category' column", csv_file
                )
                continue

            dataframe = dataframe.copy()
            if "sample_values" in dataframe.columns:
                dataframe["sample_values"] = dataframe["sample_values"].apply(self._parse_sample_values)
            semantics[dataset_name] = dataframe
            logger.debug("Loaded semantics for '%s' with %d columns", dataset_name, len(dataframe))

        if not semantics:
            raise ValueError(f"No valid semantics CSV files found in {directory}")
        return semantics

    def create_semantic_mappings(
        self,
        semantics_dir: str,
        similarity_threshold: float = 0.75,
    ) -> List[Path]:
        datasets = self.load_semantics(semantics_dir)
        dataset_names = sorted(datasets.keys())
        if len(dataset_names) < 2:
            raise ValueError("Need at least two datasets with semantics to build mappings")

        saved_paths: List[Path] = []
        for dataset_a, dataset_b in combinations(dataset_names, 2):
            mapping_df = self._build_pair_mapping(
                dataset_a,
                datasets[dataset_a],
                dataset_b,
                datasets[dataset_b],
                similarity_threshold,
            )
            if mapping_df.empty:
                logger.info(
                    "No semantic matches found between '%s' and '%s' at threshold %.2f",
                    dataset_a,
                    dataset_b,
                    similarity_threshold,
                )
                continue

            filename = f"{self._safe_name(dataset_a)}__vs__{self._safe_name(dataset_b)}.csv"
            output_path = self.output_dir / filename
            mapping_df.to_csv(output_path, index=False)
            logger.info(
                "Wrote semantic mapping for '%s' vs '%s' to %s",
                dataset_a,
                dataset_b,
                output_path,
            )
            saved_paths.append(output_path)

        if not saved_paths:
            logger.warning("No semantic mappings generated. Consider lowering the threshold.")
        return saved_paths

    def _build_pair_mapping(
        self,
        dataset_a: str,
        df_a: pd.DataFrame,
        dataset_b: str,
        df_b: pd.DataFrame,
        similarity_threshold: float,
    ) -> pd.DataFrame:
        source_terms = df_a["column_name"].tolist()
        target_terms = df_b["column_name"].tolist()
        matches = self.find_semantic_similarities(source_terms, target_terms, similarity_threshold)
        if not matches:
            return pd.DataFrame()

        meta_a = df_a.set_index("column_name").to_dict("index")
        meta_b = df_b.set_index("column_name").to_dict("index")

        rows = []
        for source_col, target_col, score in matches:
            rows.append(
                {
                    "dataset_a": dataset_a,
                    "column_a": source_col,
                    "category_a": meta_a.get(source_col, {}).get("semantic_category", "unknown"),
                    "dataset_b": dataset_b,
                    "column_b": target_col,
                    "category_b": meta_b.get(target_col, {}).get("semantic_category", "unknown"),
                    "similarity_score": score,
                }
            )

        rows.sort(key=lambda item: item["similarity_score"], reverse=True)
        return pd.DataFrame(rows)

    def find_semantic_similarities(
        self,
        source_terms: List[str],
        target_terms: List[str],
        threshold: float = 0.7,
    ) -> List[Tuple[str, str, float]]:
        if not source_terms or not target_terms:
            return []

        all_terms = source_terms + target_terms
        all_embeddings = self.get_embeddings(all_terms)
        source_embeddings = all_embeddings[: len(source_terms)]
        target_embeddings = all_embeddings[len(source_terms) :]

        similarities = cosine_similarity(source_embeddings, target_embeddings)
        matches: List[Tuple[str, str, float]] = []
        for i, source_term in enumerate(source_terms):
            for j, target_term in enumerate(target_terms):
                score = similarities[i, j]
                if score >= threshold:
                    matches.append((source_term, target_term, float(score)))

        matches.sort(key=lambda item: item[2], reverse=True)
        return matches

    def get_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            raise ValueError("texts must contain at least one item")

        embeddings: List[np.ndarray] = []
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                pooled = outputs.last_hidden_state.mean(dim=1)
                embeddings.extend(pooled.cpu().numpy())
        return np.stack(embeddings)

    def _parse_sample_values(self, value: str) -> List[str]:
        if not isinstance(value, str) or not value.strip():
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
        return [value]

    def _safe_name(self, value: str) -> str:
        sanitized = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip()]
        name = "".join(sanitized).strip("_")
        return name or "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build semantic mappings from column semantics CSV files.")
    parser.add_argument(
        "semantics_dir",
        help="Directory containing column semantics CSV files (e.g. output/column_semantics).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/semantic_mappings",
        help="Directory to store generated semantic mapping CSV files.",
    )
    parser.add_argument(
        "--model-name",
        default="bert-base-uncased",
        help="Transformer model to use for column name embeddings.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for including column pairs in the mapping.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override computation device (e.g. 'cpu', 'cuda'). Defaults to auto-detect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    mapper = ColumnSimilarityMapper(
        model_name=args.model_name,
        output_dir=args.output_dir,
        device=args.device,
    )
    paths = mapper.create_semantic_mappings(args.semantics_dir, similarity_threshold=args.threshold)
    if paths:
        print("Generated semantic mapping files:")
        for path in paths:
            print(f" - {path}")
    else:
        print("No semantic mappings produced. Check log output for details.")


if __name__ == "__main__":
    main()
