from typing import cast

from .interpreter import ClusterInterpreter, InterpreterConfig

class InterpreterFactory:
    """Factory that instantiates interpreters by name using a config object.

    Supported names (case-insensitive): 'bert', 'tfidf'. The caller must
    pass an instance of the matching config dataclass.
    """

    def create(self, name: str, config: InterpreterConfig) -> ClusterInterpreter:
        """Create an instance of a supported interpreter by name.

        Example:
            factory.create('bert', bert_config)
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        key = name.strip().lower()

        if key == "bert":
            from .bert_interpreter import BertInterpreter, BertInterpreterConfig

            if not isinstance(config, BertInterpreterConfig):
                raise TypeError("config must be a BertInterpreterConfig for interpreter 'bert'")
            cfg = cast(BertInterpreterConfig, config)
            return BertInterpreter(cfg)

        if key == "tfidf":
            from .tfidf_interpreter import TfidfInterpreter, TfidfInterpreterConfig

            if not isinstance(config, TfidfInterpreterConfig):
                raise TypeError("config must be a TfidfInterpreterConfig for interpreter 'tfidf'")
            cfg = cast(TfidfInterpreterConfig, config)
            return TfidfInterpreter(cfg)

        raise ValueError(f"Unknown interpreter: {name}")
