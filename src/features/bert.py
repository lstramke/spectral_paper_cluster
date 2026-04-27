from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from umap import UMAP
from sentence_transformers import SentenceTransformer

from .feature_extractor import FeatureExtractionResult, FeatureExtractor


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

	def __init__(
		self,
		model_name: str = "NeuML/bioclinical-modernbert-base-embeddings",
		device: str = "cpu",
		batch_size: int = 8,
		normalize: bool = True,
		show_progress: bool = False,
		umap_n_components: int | None = 100,
		umap_random_state: int = 42,
	) -> None:
		self.model_name = model_name
		self.device = device
		self.batch_size = batch_size
		self.normalize = normalize
		self.show_progress = show_progress
		self.umap_n_components = umap_n_components
		self.umap_random_state = umap_random_state

		# Load SentenceTransformer (will download if needed)
		self.model = SentenceTransformer(self.model_name, device=self.device)

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

