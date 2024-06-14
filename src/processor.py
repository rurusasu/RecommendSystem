import re
from typing import List, Union

import pandas as pd
import tensorflow as tf

from src.feat import DenseFeat, SparseFeat, VarLenSparseFeat


class Preprocessor:
    def __init__(self, features: pd.DataFrame, embedding_dims: dict):
        self.embedding_dims = embedding_dims
        self.vocabularies = self.generate_vocabularies(features)
        self.lookup_layers = self.create_lookup_layers()

    def generate_vocabularies(self, features: pd.DataFrame) -> dict:
        vocabularies = {}
        for feature_name in self.embedding_dims:
            if features[
                feature_name
            ].dtype == "object" and not feature_name.startswith("interests_"):
                vocab = sorted(features[feature_name].dropna().unique())
                vocabularies[feature_name] = vocab
            elif feature_name.startswith("interests_"):
                all_interests = []
                for interests_str in features[feature_name].dropna():
                    all_interests.extend(re.split(r",\s*", interests_str))
                vocabularies[feature_name] = sorted(set(all_interests))
        return vocabularies

    def create_lookup_layers(self) -> dict:
        lookup_layers = {}
        for feature_name, vocab in self.vocabularies.items():
            if not feature_name.startswith("interests_"):
                lookup_layers[feature_name] = tf.keras.layers.StringLookup(
                    vocabulary=vocab,
                    mask_token=None,
                    num_oov_indices=1,
                    output_mode="int",
                )
            else:
                lookup_layers[feature_name] = tf.keras.layers.StringLookup(
                    vocabulary=vocab, mask_token=None, output_mode="multi_hot"
                )
        return lookup_layers

    def process(self, data: pd.DataFrame) -> dict:
        processed_data = {}
        # NaNを含むレコードを削除
        data = data.dropna(subset=self.lookup_layers.keys())

        for feature_name, lookup_layer in self.lookup_layers.items():
            if feature_name.startswith("interests_"):
                # interests_userの分割
                interests_list = (
                    data[feature_name]
                    .apply(lambda x: re.split(r",\s*", x))
                    .tolist()
                )
                max_len = max(map(len, interests_list))
                interests_padded = [
                    interests + [""] * (max_len - len(interests))
                    for interests in interests_list
                ]
                indices = lookup_layer(interests_padded)
                processed_data[feature_name] = indices
            else:
                indices = lookup_layer(data[feature_name].astype(str).values)
                indices = tf.where(
                    indices == 0, tf.zeros_like(indices), indices - 1
                )  # インデックスの範囲を修正
                processed_data[feature_name] = indices
                max_index = len(self.vocabularies[feature_name])
                if tf.reduce_any(processed_data[feature_name] >= max_index):
                    raise ValueError(
                        f"Feature '{feature_name}' contains out-of-vocabulary index."
                    )

        # salary_userおよびsalary_targetの特別な前処理
        for feature_name in ["salary_user", "salary_target"]:
            if feature_name in data.columns:
                processed_data[feature_name] = tf.convert_to_tensor(
                    data[feature_name].values, dtype=tf.float32
                )

        return processed_data

    def create_feature_columns(
        self,
    ) -> List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]]:
        feature_columns = []
        for feature_name, vocab in self.vocabularies.items():
            if feature_name.startswith("interests_"):
                feature_columns.append(
                    VarLenSparseFeat(
                        feature_name,
                        vocabulary_size=len(vocab),
                        embedding_dim=self.embedding_dims[feature_name],
                    )
                )
            else:
                feature_columns.append(
                    SparseFeat(
                        feature_name,
                        vocabulary_size=len(vocab),
                        embedding_dim=self.embedding_dims[feature_name],
                    )
                )
        for feature_name in self.embedding_dims:
            if feature_name not in self.vocabularies:
                feature_columns.append(DenseFeat(feature_name))
        return feature_columns
