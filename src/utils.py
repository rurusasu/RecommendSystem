from typing import Dict, List, Tuple, Union

import tensorflow as tf

from src.feat import DenseFeat, SparseFeat, VarLenSparseFeat


# Embedding辞書の作成
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


# 入力データから_userで終わる特徴のみを抽出
def filter_user_inputs(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    return {k: v for k, v in inputs.items() if k.endswith("_user")}


def create_sparse_embeddings(
    user_inputs: Dict[str, tf.Tensor],
    feature_columns: List[Union[SparseFeat, DenseFeat, VarLenSparseFeat]],
    embedding_dict: Dict[str, tf.keras.layers.Embedding],
    verbose: bool = False,
) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    if verbose:
        print(
            "User Inputs:", user_inputs.keys()
        )  # デバッグ用にuser_inputsのキーを表示
        print(
            "Embedding Dict:", embedding_dict.keys()
        )  # デバッグ用にembedding_dictのキーを表示

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)
    )
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)
    )
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)
    )

    if verbose:
        print(
            "Sparse Feature Columns:",
            [feat.name for feat in sparse_feature_columns],
        )  # デバッグ用出力
        print(
            "Dense Feature Columns:",
            [feat.name for feat in dense_feature_columns],
        )  # デバッグ用出力
        print(
            "VarLen Sparse Feature Columns:",
            [feat.name for feat in varlen_sparse_feature_columns],
        )  # デバッグ用出力

    sparse_embedding_list = [
        tf.reduce_mean(
            embedding_dict[feat.name](user_inputs[feat.name]), axis=1
        )
        if len(user_inputs[feat.name].shape) == 2
        else embedding_dict[feat.name](user_inputs[feat.name])
        for feat in sparse_feature_columns
        if feat.name in user_inputs
    ]
    varlen_sparse_embedding_list = [
        tf.reduce_mean(
            embedding_dict[feat.name](user_inputs[feat.name]), axis=1
        )
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
