from clustering.base import ClusteringConfig
from clustering.clusterer_factory import ClustererFactory
from config_reader.config_reader_new import CombinedConfig
from evaluation.evaluator import ClusterEvaluator
from features.feature_extractor import FeatureConfig
from features.feature_extractor_factory import FeatureExtractorFactory
from interpretation.interpreter import InterpreterConfig
from interpretation.interpreter_factory import InterpreterFactory
from pipelines.pipeline import ExperimentPipeline, PipelineConfig

class PipelineBuilder:
    """Builder that creates a generalized ExperimentPipeline from CombinedConfig.
    
    Orchestrates the three factories:
    - FeatureExtractorFactory
    - ClustererFactory
    - InterpreterFactory
    
    To create a ready-to-run pipeline.
    """

    def __init__(
        self,
        feature_factory: FeatureExtractorFactory,
        clusterer_factory: ClustererFactory,
        interpreter_factory: InterpreterFactory,
        evaluator: ClusterEvaluator
    ) -> None:
        self.feature_factory = feature_factory
        self.clusterer_factory = clusterer_factory
        self.interpreter_factory = interpreter_factory
        self.evaluator = evaluator

    def build(self, combined_config: CombinedConfig) -> ExperimentPipeline:
        """Build a pipeline from parsed experiment configuration.
        
        Steps:
        1. Detect feature type and config from combined_config
        2. Create FeatureExtractor via feature_factory
        3. Detect clustering type and config
        4. Detect interpreter (optional) and config
        5. Assemble PipelineConfig
        6. Return ExperimentPipeline
        """

        feature_type, feature_config = self._detect_and_get_feature(combined_config)
        feature_extractor = self.feature_factory.create(feature_type, feature_config)

        clustering_type, clustering_config = self._detect_and_get_clustering(combined_config)

        interpreter = None
        interpreter_type, interpreter_config = self._detect_and_get_interpreter(combined_config)
        if interpreter_type is not None and interpreter_config is not None:
            interpreter = self.interpreter_factory.create(interpreter_type, interpreter_config)
        
        pipeline_config = PipelineConfig(
            feature_extractor=feature_extractor,
            clusterer_factory=self.clusterer_factory,
            clusterer_name=clustering_type,
            clusterer_config=clustering_config,
            evaluator=self.evaluator,
            interpreter=interpreter,
            metadata={"experiment_name": combined_config.experiment_name or "unnamed"},
        )

        return ExperimentPipeline(pipeline_config) 
    

    @staticmethod
    def _detect_and_get_feature(config: CombinedConfig) -> tuple[str, FeatureConfig]:
        """Detect feature type and return (type_name, config)."""
        if config.bert is not None:
            return "bert", config.bert
        if config.tfidf is not None:
            return "tfidf", config.tfidf
        if config.fasttext is not None:
            return "fasttext", config.fasttext
        raise ValueError("No feature config found!")

    @staticmethod
    def _detect_and_get_clustering(config: CombinedConfig) -> tuple[str, ClusteringConfig]:
        """Detect clustering type and return (type_name, config)."""
        if config.kmeans is not None:
            return "kmeans", config.kmeans
        if config.spectral is not None:
            return "spectral", config.spectral
        if config.dbscan is not None:
            return "dbscan", config.dbscan
        if config.optics is not None:
            return "optics", config.optics
        if config.agglomerative is not None:
            return "agglomerative", config.agglomerative
        if config.affinityPropagation is not None:
            return "affinity_propagation", config.affinityPropagation
        if config.gaussianMixture is not None:
            return "gmm", config.gaussianMixture
        if config.hdbscan is not None:
            return "hdbscan", config.hdbscan
        raise ValueError("No clustering config found!")
    
    @staticmethod
    def _detect_and_get_interpreter(config: CombinedConfig) -> tuple[str, InterpreterConfig] | tuple[None, None]:
        """Detect interpreter type and return (type_name, config) or (None, None) if no interpreter."""
        if config.interpretation_bert is not None:
            return "bert", config.interpretation_bert
        if config.interpretation is not None:
            return "tfidf", config.interpretation
        return None, None
