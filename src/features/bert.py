from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertConfig
from umap import UMAP
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os


def _load_sentence_transformer(model_name: str, device: str) -> SentenceTransformer:
	"""Load a SentenceTransformer model, preferring a local cache under
	`models/<model_slug>` (or directory from `LOCAL_MODELS_DIR`). If the
	local copy doesn't exist, attempt to download and save it; fall back to
	the normal loader on failure.
	"""
	model_source = model_name
	# Always use the project's ./models directory for caching
	local_models_dir = Path("models")
	# If user provided a local path, prefer it
	if Path(model_name).exists():
		model_source = model_name
	else:
		model_slug = model_name.replace("/", "_")
		local_copy = local_models_dir / model_slug
		if local_copy.exists():
			model_source = str(local_copy)
		else:
			# download and save into local_models_dir (best-effort)
			try:
				print(f"Downloading SentenceTransformer '{model_name}' and caching to {local_copy}...")
				model = SentenceTransformer(model_name, device=device)
				try:
					local_copy.parent.mkdir(parents=True, exist_ok=True)
					model.save(str(local_copy))
					print(f"Saved model to {local_copy}")
				except Exception:
					print("Warning: failed to save local model cache; continuing with in-memory model.")
				return model
			except Exception:
				print(f"Warning: failed to download model '{model_name}' from HF Hub. Will retry via SentenceTransformer loader.")
				model_source = model_name

	# Load model from chosen source (local dir or HF id)
	return SentenceTransformer(model_source, device=device)

from .feature_extractor import FeatureExtractionResult, FeatureExtractor

@dataclass(slots=True)
class BERTConfig:
	model_name: str
	device: str
	batch_size: int
	normalize: bool
	show_progress: bool
	umap_n_components: int | None
	umap_random_state: int

class BertFeatureExtractor(FeatureExtractor):
	"""Extract sentence/document embeddings using a Sentence‑Transformers model.

	Returns features as a L2‑normalized `torch.Tensor` and metadata following
	the project's `FeatureExtractionResult` contract.

	Parameters
	- model_name: HF model id (default: NeuML bioclinical modernbert embeddings)
	- device: 'cpu' or 'cuda'
	- batch_size: encode batch size
	- normalize: whether to L2 normalize output embeddings
	- show_progress: show progress bar during encoding
	"""

	def __init__(self, config: BERTConfig) -> None:
		self.model_name = config.model_name
		self.device = config.device
		self.batch_size = config.batch_size
		self.normalize = config.normalize
		self.show_progress = config.show_progress
		self.umap_n_components = config.umap_n_components
		self.umap_random_state = config.umap_random_state

		# Load SentenceTransformer via helper (prefers local cache under ./models/)
		self.model = _load_sentence_transformer(self.model_name, self.device)

	def extract_features(self, documents: list[str]) -> FeatureExtractionResult:
		if not documents:
			raise ValueError("documents must not be empty")

		emb_tensor = self.model.encode(
			documents,
			batch_size=self.batch_size,
			show_progress_bar=self.show_progress,
			convert_to_tensor=True,
		)

		if emb_tensor.device.type != "cpu":
			emb_tensor = emb_tensor.cpu()

		if self.normalize:
			emb_tensor = F.normalize(emb_tensor, p=2, dim=1)

		# Preserve original embeddings in case UMAP reduction is applied
		original_tensor = emb_tensor

		# Optionally apply UMAP reduction
		if self.umap_n_components is not None and original_tensor.ndim == 2:
			emb_np = original_tensor.numpy()
			umap_reducer = UMAP(n_components=self.umap_n_components, random_state=self.umap_random_state)
			reduced_np = umap_reducer.fit_transform(emb_np)
			emb_tensor = torch.from_numpy(np.asarray(reduced_np)).float()

		dim = emb_tensor.size(1) if emb_tensor.ndim == 2 else 0
		if self.umap_n_components is not None:
			feature_names = [f"bert_umap_{i}" for i in range(dim)]
		else:
			feature_names = [f"bert_{i}" for i in range(dim)]

		metadata = {
			"extractor": "bert",
			"model_name": self.model_name,
			"device": self.device,
			"batch_size": self.batch_size,
			"embedding_dim": dim,
			"normalized": bool(self.normalize),
			"sentence_transformers_version": getattr(self.model, "__version__", None),
			"raw_documents": documents
		}
		if self.umap_n_components is not None:
			metadata.update({"umap_n_components": self.umap_n_components, "umap_random_state": self.umap_random_state})

		return FeatureExtractionResult(
			features=emb_tensor,
			feature_names=feature_names,
			original_features=original_tensor,
			original_feature_names=[f"bert_{i}" for i in range(original_tensor.size(1))] if original_tensor.ndim == 2 else [],
			metadata=metadata,
		)

