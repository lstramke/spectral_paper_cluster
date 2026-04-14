from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, fields

import yaml

from clustering.agglomerativeClustering import AgglomerativeConfig
from configs.config_reader.agglomerative_config_reader import AgglomerativeConfigReader
from configs.config_reader.affinityPropagation_config_reader import AffinityPropagationConfigReader
from src.clustering.affinityPropagation import AffinityPropagationConfig

from .tfidf_config_reader import TfidfConfigReader
from .dbscan_config_reader import DbscanConfigReader
from .interpretation_config_reader import InterpretationConfigReader
from .kmeans_config_reader import KMeansConfigReader
from .optics_config_reader import OpticsConfigReader
from .hdbscan_config_reader import HdbscanConfigReader
from .input_config_reader import InputConfigReader, InputConfig
from .output_config_reader import OutputsConfigReader, OutputsConfig
from src.clustering.dbscan import DBSCANConfig
from src.clustering.kmeans import KMeansConfig
from src.clustering.optics import OpticsConfig
from src.clustering.hdbscan import HDBSCANConfig
from src.features.tfidf import TfidfConfig
from src.interpretation.tfidf_interpreter import TfidfInterpreterConfig

@dataclass(slots=True)
class RegisteredReaders:
	tfidf: Optional[TfidfConfigReader] = None
	dbscan: Optional[DbscanConfigReader] = None
	interpretation: Optional[InterpretationConfigReader] = None
	kmeans: Optional[KMeansConfigReader] = None
	optics: Optional[OpticsConfigReader] = None
	hdbscan: Optional[HdbscanConfigReader] = None
	agglomerative: Optional[AgglomerativeConfigReader] = None	
	affinityPropagation: Optional[AffinityPropagationConfigReader] = None
	input: Optional[InputConfigReader] = None
	outputs: Optional[OutputsConfigReader] = None

	def __iter__(self):
		"""Iterate over (name, reader) pairs for all dataclass fields."""
		for f in fields(self):
			yield f.name, getattr(self, f.name)


@dataclass(slots=True)
class CombinedConfig:
	experiment_name: Optional[str]
	input: Optional[InputConfig]
	kmeans: Optional[KMeansConfig]
	dbscan: Optional[DBSCANConfig]
	optics: Optional[OpticsConfig]
	hdbscan: Optional[HDBSCANConfig]
	agglomerative: Optional[AgglomerativeConfig]
	affinityPropagation: Optional[AffinityPropagationConfig]
	tfidf: Optional[TfidfConfig]
	interpretation: Optional[TfidfInterpreterConfig]
	outputs: Optional[OutputsConfig]

class ConfigReader:
	"""A concrete config reader produced by `ConfigReaderBuilder.build()`.

	Provides a single `read(config_path)` method that loads the YAML once
	and dispatches to the registered section readers. The reader obtains the
	registered readers from the originating `ConfigReaderBuilder` instance.
	"""

	def __init__(self, builder: ConfigReaderBuilder) -> None:
		self._readers = RegisteredReaders(
			input=builder._registered.input,
			kmeans=builder._registered.kmeans,
			dbscan=builder._registered.dbscan,
			optics=builder._registered.optics,
			hdbscan=builder._registered.hdbscan,
			agglomerative=builder._registered.agglomerative,
			affinityPropagation=builder._registered.affinityPropagation,
			tfidf=builder._registered.tfidf,
			interpretation=builder._registered.interpretation,
			outputs=builder._registered.outputs,
		)

	def read(self, config_path: Path) -> CombinedConfig:
		with config_path.open("r", encoding="utf-8") as fh:
			raw = yaml.safe_load(fh)

		if raw is None:
			raise ValueError("Config is empty")
		if not isinstance(raw, dict):
			raise ValueError("Config root must be a mapping")

		results: Dict[str, Any] = {}
		for name, reader in self._readers:
			if reader is not None:
				results[name] = reader.read_section(raw)

		experiment_name: Optional[str] = None
		if "experiment_name" in raw and raw.get("experiment_name") is not None:
			experiment_name = str(raw.get("experiment_name"))

		return CombinedConfig(
			experiment_name=experiment_name,
			input=results.get("input"),
			kmeans=results.get("kmeans"),
			dbscan=results.get("dbscan"),
			optics=results.get("optics"),
			hdbscan=results.get("hdbscan"),
			agglomerative=results.get("agglomerative"),
			affinityPropagation=results.get("affinityPropagation"),
			tfidf=results.get("tfidf"),
			interpretation=results.get("interpretation"),
			outputs=results.get("outputs"),
		)

class ConfigReaderBuilder:
	"""Builder for modular config readers."""

	def __init__(self) -> None:
		self._registered = RegisteredReaders()

	def add_tfidf(self) -> ConfigReaderBuilder:
		self._registered.tfidf = TfidfConfigReader()
		return self

	def add_dbscan(self) -> ConfigReaderBuilder:
		self._registered.dbscan = DbscanConfigReader()
		return self

	def add_kmeans(self) -> ConfigReaderBuilder:
		self._registered.kmeans = KMeansConfigReader()
		return self

	def add_optics(self) -> ConfigReaderBuilder:
		self._registered.optics = OpticsConfigReader()
		return self
	
	def add_hdbscan(self) -> ConfigReaderBuilder:
		self._registered.hdbscan = HdbscanConfigReader()
		return self

	def add_agglomerative(self) -> ConfigReaderBuilder:
		self._registered.agglomerative = AgglomerativeConfigReader()
		return self

	def add_affinityPropagation(self) -> ConfigReaderBuilder:
		self._registered.affinityPropagation = AffinityPropagationConfigReader()
		return self

	def add_interpretation(self) -> ConfigReaderBuilder:
		self._registered.interpretation = InterpretationConfigReader()
		return self

	def add_input(self) -> ConfigReaderBuilder:
		self._registered.input = InputConfigReader()
		return self

	def add_outputs(self) -> ConfigReaderBuilder:
		self._registered.outputs = OutputsConfigReader()
		return self

	def build(self) -> ConfigReader:
		return ConfigReader(self)
