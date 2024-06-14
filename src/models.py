import json

import tensorflow as tf

from src.feat import DenseFeat, SparseFeat
from src.layers import DNN, LightSE
from src.utils import create_sparse_embeddings


class UserModel(tf.keras.Model):
    def __init__(
        self,
        layer_sizes,
        field_size: int,
        embedding_size: int,
        embedding_dict: dict,
        feature_columns,
        activation: str = "relu",
        use_bn: bool = False,
        use_ln: bool = False,
        verbose: bool = False,
        name=None,  # nameパラメータを追加
        trainable=True,
        dtype=tf.float32,
    ):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self.User_SE = LightSE(
            field_size=field_size, embedding_size=embedding_size
        )
        self.feature_columns = feature_columns
        self.embedding_dict = embedding_dict
        self.dnn = DNN(
            layer_sizes=layer_sizes,
            activation=activation,
            use_bn=use_bn,
            use_ln=use_ln,
        )
        self.final_dense = tf.keras.layers.Dense(1)
        self.flatten = tf.keras.layers.Flatten()
        self.verbose = verbose

    def build(self, input_shape):
        super(UserModel, self).build(input_shape)

    def call(self, inputs: dict) -> tf.Tensor:
        sparse_embeddings, dense_values = create_sparse_embeddings(
            inputs, self.feature_columns, self.embedding_dict, self.verbose
        )
        user_dnn_input = tf.concat(sparse_embeddings, axis=-1)
        User_sim_embedding = self.User_SE(
            tf.expand_dims(user_dnn_input, axis=1)
        )
        sparse_dnn_input = self.flatten(User_sim_embedding)
        dense_values_flat = tf.concat(
            [
                tf.expand_dims(d, axis=-1) if len(d.shape) == 1 else d
                for d in dense_values
            ],
            axis=-1,
        )
        user_dnn_input = tf.concat(
            [sparse_dnn_input, dense_values_flat], axis=1
        )
        dnn_output = self.dnn(user_dnn_input)
        final_output = self.final_dense(dnn_output)
        return final_output

    def get_config(self):
        config = super(UserModel, self).get_config()
        config.update(
            {
                "layer_sizes": self.dnn.layer_sizes,
                "field_size": self.User_SE.field_size,
                "embedding_size": self.User_SE.embedding_size,
                "embedding_dict": json.dumps(
                    {k: v.get_config() for k, v in self.embedding_dict.items()}
                ),
                "feature_columns": json.dumps(
                    [feat.get_config() for feat in self.feature_columns]
                ),
                "activation": self.dnn.activation,
                "use_bn": self.dnn.use_bn,
                "use_ln": self.dnn.use_ln,
                "verbose": self.verbose,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        embedding_dict_config = json.loads(config.pop("embedding_dict"))
        embedding_dict = {
            k: tf.keras.layers.Embedding.from_config(v)
            for k, v in embedding_dict_config.items()
        }
        feature_columns_config = json.loads(config.pop("feature_columns"))
        feature_columns = [
            SparseFeat.from_config(feat)
            if "embedding_dim" in feat
            else DenseFeat.from_config(feat)
            for feat in feature_columns_config
        ]
        return cls(
            embedding_dict=embedding_dict,
            feature_columns=feature_columns,
            **config,
        )


class ItemModel(tf.keras.Model):
    def __init__(
        self,
        layer_sizes,
        field_size: int,
        embedding_size: int,
        embedding_dict: dict,
        feature_columns,
        activation: str = "relu",
        use_bn: bool = False,
        use_ln: bool = False,
        verbose: bool = False,
        name=None,  # nameパラメータを追加
        trainable=True,
        dtype=tf.float32,
    ):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self.Item_SE = LightSE(
            field_size=field_size, embedding_size=embedding_size
        )
        self.embedding_dict = embedding_dict
        self.feature_columns = feature_columns
        self.dnn = DNN(
            layer_sizes=layer_sizes,
            activation=activation,
            use_bn=use_bn,
            use_ln=use_ln,
        )
        self.final_dense = tf.keras.layers.Dense(1)
        self.flatten = tf.keras.layers.Flatten()
        self.verbose = verbose

    def build(self, input_shape):
        super(ItemModel, self).build(input_shape)

    def call(self, inputs: dict) -> tf.Tensor:
        sparse_embeddings, dense_values = create_sparse_embeddings(
            inputs, self.feature_columns, self.embedding_dict, self.verbose
        )
        item_dnn_input = tf.concat(sparse_embeddings, axis=-1)
        Item_sim_embedding = self.Item_SE(
            tf.expand_dims(item_dnn_input, axis=1)
        )
        sparse_dnn_input = self.flatten(Item_sim_embedding)
        dense_values_flat = tf.concat(
            [
                tf.expand_dims(d, axis=-1) if len(d.shape) == 1 else d
                for d in dense_values
            ],
            axis=-1,
        )
        item_input = tf.concat([sparse_dnn_input, dense_values_flat], axis=1)
        dnn_output = self.dnn(item_input)
        final_output = self.final_dense(dnn_output)
        return final_output

    def get_config(self):
        config = super(ItemModel, self).get_config()
        config.update(
            {
                "layer_sizes": self.dnn.layer_sizes,
                "field_size": self.Item_SE.field_size,
                "embedding_size": self.Item_SE.embedding_size,
                "embedding_dict": json.dumps(
                    {k: v.get_config() for k, v in self.embedding_dict.items()}
                ),
                "feature_columns": json.dumps(
                    [feat.get_config() for feat in self.feature_columns]
                ),
                "activation": self.dnn.activation,
                "use_bn": self.dnn.use_bn,
                "use_ln": self.dnn.use_ln,
                "verbose": self.verbose,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        embedding_dict_config = json.loads(config.pop("embedding_dict"))
        embedding_dict = {
            k: tf.keras.layers.Embedding.from_config(v)
            for k, v in embedding_dict_config.items()
        }
        feature_columns_config = json.loads(config.pop("feature_columns"))
        feature_columns = [
            SparseFeat.from_config(feat)
            if "embedding_dim" in feat
            else DenseFeat.from_config(feat)
            for feat in feature_columns_config
        ]
        return cls(
            embedding_dict=embedding_dict,
            feature_columns=feature_columns,
            **config,
        )


class TwoTowerModel(tf.keras.Model):
    def __init__(
        self,
        layer_sizes,
        field_size: int,
        embedding_size: int,
        embedding_dict,
        feature_columns,
        activation: str = "relu",
        use_bn: bool = False,
        use_ln: bool = False,
        verbose: bool = False,
        name=None,  # nameパラメータを追加
        trainable=True,
        dtype=tf.float32,
    ):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self.user_model = UserModel(
            layer_sizes,
            field_size,
            embedding_size,
            embedding_dict,
            feature_columns=feature_columns,
            activation=activation,
            use_bn=use_bn,
            use_ln=use_ln,
            verbose=verbose,
        )
        self.item_model = ItemModel(
            layer_sizes,
            field_size,
            embedding_size,
            embedding_dict,
            feature_columns=feature_columns,
            activation=activation,
            use_bn=use_bn,
            use_ln=use_ln,
            verbose=verbose,
        )
        self.verbose = verbose

    def build(self, input_shape):
        self.user_model.build(input_shape)  # ユーザーモデルの構築
        self.item_model.build(input_shape)  # アイテムモデルの構築
        super().build(input_shape)  # 必須です

    def call(self, inputs):
        user_inputs = {
            key: value for key, value in inputs.items() if key.endswith("_user")
        }
        target_inputs = {
            key: value
            for key, value in inputs.items()
            if key.endswith("_target")
        }

        user_embeddings = self.user_model(user_inputs)
        target_embeddings = self.item_model(target_inputs)

        if self.verbose:
            tf.print("ユーザー埋め込みの形状:", tf.shape(user_embeddings))
            tf.print("ターゲット埋め込みの形状:", tf.shape(target_embeddings))

        user_batch_size = user_embeddings.shape[0]
        target_batch_size = target_embeddings.shape[0]

        if user_batch_size != target_batch_size:
            raise ValueError(
                f"ユーザーとターゲットの埋め込みのバッチサイズは同じでなければなりませんが、{user_batch_size} と {target_batch_size} が与えられました。"
            )

        user_norm = tf.nn.l2_normalize(user_embeddings, axis=-1)
        target_norm = tf.nn.l2_normalize(target_embeddings, axis=-1)
        similarity = tf.reduce_sum(
            tf.multiply(user_norm, target_norm), axis=-1, keepdims=True
        )

        return similarity

    def get_config(self):
        config = super(TwoTowerModel, self).get_config()
        config.update(
            {
                "layer_sizes": self.user_model.dnn.layer_sizes,
                "field_size": self.user_model.User_SE.field_size,
                "embedding_size": self.user_model.User_SE.embedding_size,
                "embedding_dict": json.dumps(
                    {
                        k: v.get_config()
                        for k, v in self.user_model.embedding_dict.items()
                    }
                ),
                "feature_columns": json.dumps(
                    [
                        feat.get_config()
                        for feat in self.user_model.feature_columns
                    ]
                ),
                "activation": self.user_model.dnn.activation,
                "use_bn": self.user_model.dnn.use_bn,
                "use_ln": self.user_model.dnn.use_ln,
                "verbose": self.verbose,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        embedding_dict_config = json.loads(config.pop("embedding_dict"))
        embedding_dict = {
            k: tf.keras.layers.Embedding.from_config(v)
            for k, v in embedding_dict_config.items()
        }
        feature_columns_config = json.loads(config.pop("feature_columns"))
        feature_columns = [
            SparseFeat.from_config(feat)
            if "embedding_dim" in feat
            else DenseFeat.from_config(feat)
            for feat in feature_columns_config
        ]
        return cls(
            embedding_dict=embedding_dict,
            feature_columns=feature_columns,
            **config,
        )
