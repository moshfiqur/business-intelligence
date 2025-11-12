"""BERT-based semantic extractor for enterprise datasets."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


class EnterpriseSemanticExtractor:
    """Understand semantic similarity between enterprise column names."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.logger = logging.getLogger(__name__)

    def get_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Generate embeddings for a list of texts using mean-pooled hidden states."""
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
                # mean pool the last hidden state to obtain sentence embeddings
                pooled = outputs.last_hidden_state.mean(dim=1)
                embeddings.extend(pooled.cpu().numpy())

        return np.stack(embeddings)

    def find_semantic_similarities(
        self,
        source_terms: List[str],
        target_terms: List[str],
        threshold: float = 0.7,
    ) -> List[Tuple[str, str, float]]:
        """Return term pairs whose cosine similarity exceeds the given threshold."""
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

    def extract_column_semantics(self, dataframe: pd.DataFrame) -> Dict[str, Dict]:
        """Capture metadata about each column to support semantic mapping."""
        semantics: Dict[str, Dict] = {}
        for column in dataframe.columns:
            semantics[column] = {
                "column_name": column,
                "data_type": str(dataframe[column].dtype),
                "sample_values": self._get_sample_values(dataframe[column]),
                "semantic_category": self._infer_semantic_category(column, dataframe[column]),
            }
        return semantics

    def _get_sample_values(self, series: pd.Series, n_samples: int = 5) -> List[str]:
        """Return representative sample values without failing on short series."""
        clean_series = series.dropna()
        if clean_series.empty:
            return []

        unique_values = clean_series.unique().tolist()
        if len(unique_values) <= n_samples:
            return [str(value) for value in unique_values]

        frequent = clean_series.value_counts().index.tolist()[: max(1, n_samples // 2)]
        remaining_needed = n_samples - len(frequent)
        remainder_pool = clean_series[~clean_series.isin(frequent)]
        if remainder_pool.empty:
            random_samples = []
        else:
            random_samples = remainder_pool.sample(
                n=remaining_needed,
                replace=len(remainder_pool) < remaining_needed,
                random_state=42,
            ).tolist()

        samples = (frequent + random_samples)[:n_samples]
        return [str(sample) for sample in samples]

    def _infer_semantic_category(self, column_name: str, series: pd.Series) -> str:
        """Infer semantic category using heuristics from column names and dtype."""
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

    def create_semantic_mapping(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        similarity_threshold: float = 0.8,
    ) -> Dict[str, List[Dict]]:
        """Create mapping between dataframes where column semantics align."""
        if source_df.empty or target_df.empty:
            return {}

        source_semantics = self.extract_column_semantics(source_df)
        target_semantics = self.extract_column_semantics(target_df)

        similarities = self.find_semantic_similarities(
            list(source_semantics.keys()),
            list(target_semantics.keys()),
            threshold=similarity_threshold,
        )

        mapping: Dict[str, List[Dict]] = {}
        for source_col, target_col, score in similarities:
            mapping.setdefault(source_col, []).append(
                {
                    "target_column": target_col,
                    "similarity_score": score,
                    "source_category": source_semantics[source_col]["semantic_category"],
                    "target_category": target_semantics[target_col]["semantic_category"],
                }
            )
        return mapping


if __name__ == "__main__":
    extractor = EnterpriseSemanticExtractor()

    sample_data_1 = pd.DataFrame(
        {
            "employee_id": [101, 102, 103],
            "emp_name": ["John Doe", "Jane Smith", "Bob Johnson"],
            "salary_amount": [50000, 60000, 55000],
            "hire_date": ["2020-01-15", "2019-03-20", "2021-06-10"],
        }
    )

    sample_data_2 = pd.DataFrame(
        {
            "staff_id": [201, 202, 203],
            "staff_name": ["Alice Brown", "Charlie Wilson", "Diana Davis"],
            "compensation": [70000, 65000, 72000],
            "start_date": ["2018-11-05", "2020-02-14", "2019-09-30"],
        }
    )

    semantics_1 = extractor.extract_column_semantics(sample_data_1)
    semantics_2 = extractor.extract_column_semantics(sample_data_2)
    mapping = extractor.create_semantic_mapping(sample_data_1, sample_data_2)

    print("Dataset 1 semantics:")
    for col, info in semantics_1.items():
        print(f"  {col}: {info['semantic_category']}")

    print("\nDataset 2 semantics:")
    for col, info in semantics_2.items():
        print(f"  {col}: {info['semantic_category']}")

    print("\nSemantic mapping suggestions:")
    for source_col, matches in mapping.items():
        for match in matches:
            print(
                f"  {source_col} -> {match['target_column']} "
                f"(score: {match['similarity_score']:.3f}, "
                f"{match['source_category']}→{match['target_category']})"
            )
