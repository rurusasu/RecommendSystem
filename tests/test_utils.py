from typing import Dict, List, Tuple, Union

import pytest
import tensorflow as tf

from src.feat import DenseFeat, SparseFeat, VarLenSparseFeat


# エンベディング辞書の作成関数
def create_embedding_dict(
    feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]],
) -> Dict[str, tf.keras.layers.Embedding]:
    embedding_dict = {}
    for feat in feature_columns:
        if isinstance(feat, SparseFeat) or isinstance(feat, VarLenSparseFeat):
            embedding_dict[feat.name] = tf.keras.layers.Embedding(
                feat.vocabulary_size, feat.embedding_dim
            )
    return embedding_dict


# 入力データから_userで終わる特徴のみを抽出する関数
def filter_user_inputs(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    return {k: v for k, v in inputs.items() if k.endswith("_user")}


# VarLenSparseFeat のエンベディングを平均化して形状を変換する関数
def create_varlen_sparse_embedding(
    embedding_dict: Dict[str, tf.keras.layers.Embedding],
    user_inputs: Dict[str, tf.Tensor],
    feature: VarLenSparseFeat,
) -> tf.Tensor:
    embedding = embedding_dict[feature.name](user_inputs[feature.name])
    embedding_mean = tf.reduce_mean(embedding, axis=1)
    return embedding_mean


# スパース特徴量のエンベディングの作成関数
def create_sparse_embeddings(
    user_inputs: Dict[str, tf.Tensor],
    feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]],
    embedding_dict: Dict[str, tf.keras.layers.Embedding],
) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)
    )
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)
    )
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)
    )

    sparse_embedding_list = [
        embedding_dict[feat.name](user_inputs[feat.name])
        for feat in sparse_feature_columns
        if feat.name in user_inputs
    ]
    varlen_sparse_embedding_list = [
        create_varlen_sparse_embedding(embedding_dict, user_inputs, feat)
        for feat in varlen_sparse_feature_columns
        if feat.name in user_inputs
    ]
    dense_value_list = [
        user_inputs[feat.name]
        for feat in dense_feature_columns
        if feat.name in user_inputs
    ]

    return (
        sparse_embedding_list + varlen_sparse_embedding_list,
        dense_value_list,
    )


# テストケース
def test_create_embedding_dict():
    feature_columns = [
        SparseFeat("sparse_feat1", 10, 4),
        SparseFeat("sparse_feat2", 15, 4),
        VarLenSparseFeat("varlen_feat1", 20, 4),
        DenseFeat("dense_feat1"),
    ]
    embedding_dict = create_embedding_dict(feature_columns)
    assert len(embedding_dict) == 3
    assert "sparse_feat1" in embedding_dict
    assert "sparse_feat2" in embedding_dict
    assert "varlen_feat1" in embedding_dict
    assert isinstance(embedding_dict["sparse_feat1"], tf.keras.layers.Embedding)
    assert isinstance(embedding_dict["sparse_feat2"], tf.keras.layers.Embedding)
    assert isinstance(embedding_dict["varlen_feat1"], tf.keras.layers.Embedding)


def test_filter_user_inputs():
    inputs = {
        "feature1_user": tf.constant([1, 2, 3]),
        "feature2_user": tf.constant([4, 5, 6]),
        "feature3_item": tf.constant([7, 8, 9]),
    }
    filtered_inputs = filter_user_inputs(inputs)
    assert len(filtered_inputs) == 2
    assert "feature1_user" in filtered_inputs
    assert "feature2_user" in filtered_inputs
    assert "feature3_item" not in filtered_inputs


def test_create_varlen_sparse_embedding():
    embedding_layer = tf.keras.layers.Embedding(20, 4)
    embedding_dict = {"varlen_feat1": embedding_layer}
    user_inputs = {"varlen_feat1": tf.constant([[1, 2], [3, 4], [5, 6]])}
    feature = VarLenSparseFeat("varlen_feat1", 20, 4)
    varlen_embedding = create_varlen_sparse_embedding(
        embedding_dict, user_inputs, feature
    )
    assert varlen_embedding.shape == (3, 4)


def test_create_sparse_embeddings():
    embedding_dict = {
        "sparse_feat1": tf.keras.layers.Embedding(10, 4),
        "varlen_feat1": tf.keras.layers.Embedding(20, 4),
    }
    user_inputs = {
        "sparse_feat1": tf.constant([1, 2, 3]),
        "varlen_feat1": tf.constant([[1, 2], [3, 4], [5, 6]]),
        "dense_feat1": tf.constant([[1.0], [2.0], [3.0]]),
    }
    feature_columns = [
        SparseFeat("sparse_feat1", 10, 4),
        VarLenSparseFeat("varlen_feat1", 20, 4),
        DenseFeat("dense_feat1"),
    ]
    sparse_embeddings, dense_values = create_sparse_embeddings(
        user_inputs, feature_columns, embedding_dict
    )
    assert len(sparse_embeddings) == 2
    assert len(dense_values) == 1
    assert sparse_embeddings[0].shape == (3, 4)
    assert sparse_embeddings[1].shape == (3, 4)
    assert dense_values[0].shape == (3, 1)


if __name__ == "__main__":
    pytest.main()
