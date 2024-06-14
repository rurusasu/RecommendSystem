import tensorflow as tf


class SparseFeat:
    def __init__(
        self,
        name,
        vocabulary_size,
        embedding_dim,
        dtype=tf.int32,
        use_hash=False,
    ):
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.use_hash = use_hash

    def get_config(self):
        return {
            "name": self.name,
            "vocabulary_size": self.vocabulary_size,
            "embedding_dim": self.embedding_dim,
            "dtype": self.dtype.name,
            "use_hash": self.use_hash,
        }

    @classmethod
    def from_config(cls, config):
        config["dtype"] = tf.dtypes.as_dtype(config.get("dtype", "int32"))
        return cls(**config)


class DenseFeat:
    def __init__(self, name: str):
        self.name = name

    def get_config(self):
        return {"name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class VarLenSparseFeat:
    def __init__(self, name: str, vocabulary_size: int, embedding_dim: int):
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim

    def get_config(self):
        return {
            "name": self.name,
            "vocabulary_size": self.vocabulary_size,
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
