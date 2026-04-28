from __future__ import annotations

from dataclasses import dataclass

import torch
from typing import Optional

from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sentence_transformers import SentenceTransformer
from pathlib import Path

from clustering.base import ClusteringResult
from features.feature_extractor import FeatureExtractionResult
from .interpreter import ClusterInterpreter, InterpretationResult


@dataclass(slots=True)
class BertInterpreterConfig:
	top_n_terms: int = 10
	max_features: Optional[int] = None
	model_name: Optional[str] = None
	spacy_pipeline: str = "en_core_web_sm"
	pos_pattern: str = "<ADJ.*>*<N.*>+"
	use_mmr: bool = False
	diversity: float = 0.5
	nr_candidates: Optional[int] = None


class BertInterpreter(ClusterInterpreter):
	"""Interpreter using KeyBERT + KeyphraseVectorizers per-cluster.

	Parameters
	- config: BertInterpreterConfig (model_name must be set to use a specific SBERT)
	"""

	def __init__(self, config: BertInterpreterConfig) -> None:
		self.config = config
		# lazy-loaded SentenceTransformer instance (loaded from config.model_name)
		self.sbert_model: Optional[SentenceTransformer] = None

	def interpret(self, features: FeatureExtractionResult, clustering: ClusteringResult, labels_true: torch.Tensor | None = None,) -> InterpretationResult:
		docs = features.metadata.get("raw_documents")
		if docs is None:
			raise ValueError("raw_documents not found in features.metadata; cannot compute keyphrases")
		if not docs:
			raise ValueError("raw_documents is empty")

		labels = clustering.labels.detach().cpu()

		if self.config.model_name is not None:
			if self.sbert_model is None:
				# Use only a locally available model. Accept either a direct path
				# in `model_name` or a cached model under ./models/<model_slug>.
				model_source = self.config.model_name
				if Path(self.config.model_name).exists():
					model_source = self.config.model_name
				else:
					model_slug = self.config.model_name.replace("/", "_")
					# Always use the project's ./models directory for cached models
					local_models_dir = Path("models")
					local_copy = local_models_dir / model_slug
					if local_copy.exists():
						model_source = str(local_copy)
					else:
						raise RuntimeError(
							f"Local SentenceTransformer model not found.\n"
							f"Expected a local path at '{self.config.model_name}' or a cached model at '{local_copy}'.\n"
							"Please download the model and place it there or set LOCAL_MODELS_DIR to the folder containing cached models."
						)
				# Load model from the local source
				self.sbert_model = SentenceTransformer(model_source, device="cpu")
			kw_model = KeyBERT(model=self.sbert_model) # type: ignore
		else:
			kw_model = KeyBERT()
		cluster_terms: dict[int, list[tuple[str, float]]] = {}

		for cluster_id in sorted(int(value) for value in labels.unique().tolist()):
			mask = (labels == cluster_id).numpy()
			if not mask.any():
				cluster_terms[cluster_id] = []
				continue

			cluster_docs = [docs[i] for i, flag in enumerate(mask) if flag]

			try:
				kv = KeyphraseCountVectorizer(spacy_pipeline=self.config.spacy_pipeline, pos_pattern=self.config.pos_pattern)
				# let vectorizer build candidates on the cluster docs
				try:
					kv.fit_transform(cluster_docs)
				except Exception:
					pass

				text = "\n".join(cluster_docs)
				keywords = kw_model.extract_keywords(
					text, 
					vectorizer=kv,  # pyright: ignore[reportArgumentType]
					top_n=self.config.top_n_terms, 
				)
				cluster_terms[cluster_id] = [(str(k), float(s)) for k, s in keywords] # pyright: ignore[reportArgumentType]
			except Exception:
				print(f"skipped {cluster_id}")
				cluster_terms[cluster_id] = []

		return InterpretationResult(
			cluster_terms=cluster_terms,
			metadata={
				"interpreter": "bert_keybert_keyphrase",
				"top_n_terms": self.config.top_n_terms,
				"spacy_pipeline": self.config.spacy_pipeline,
				"pos_pattern": self.config.pos_pattern,
				"use_mmr": self.config.use_mmr,
				"nr_candidates": self.config.nr_candidates,
				"labels_true_provided": labels_true is not None,
			},
		)
