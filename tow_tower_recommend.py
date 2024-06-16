import tensorflow as tf
import pickle
import pandas as pd
import tensorflow_recommenders as tfrs
from src.feat import SparseFeat
from src.models import UserModel, ItemModel, TwoTowerModel
from src.layers import LightSE, DNN
from src.processor import Preprocessor
from src.utils import create_embedding_dict

base_dir = "/home/user/core/data"

with open(f"{base_dir}/sample_merged_full_rating_conv.pkl", "rb") as f:
    data = pickle.load(f)
with open(f"{base_dir}/user_profile.pkl", "rb") as f:
    user_data = pickle.load(f)

# ラベルに不正な値が含まれている可能性があるので、削除
cleaning_data = data[data["rating"].isin([0, 1, 2])]
cleaning_data = cleaning_data.dropna()
cleaning_user_data = user_data.dropna()

# 特定のuser_idのデータをフィルタリング
def get_user_data(user_id, cleaning_user_data):
    user_data = cleaning_user_data[cleaning_user_data["user_id"] == user_id]
    #user_data = user_data.drop_duplicates(subset=["user_id"])
    return user_data

user_id = 1
user_data = get_user_data(user_id=user_id, cleaning_user_data=cleaning_user_data)

# 不要なカラムの削除
cleaning_data = cleaning_data.drop(
    columns=[
        "user_id",
        "target_id",
        "user_name_target",
        "nickname_target",
        "plan_target",
        "account_creation_timestamp_target",
        "user_name_user",
        "nickname_user",
        "plan_user",
        "account_creation_timestamp_user",
    ]
)
user_data = user_data.drop(
    columns=[
        "user_id",
        "user_name_user",
        "nickname_user",
        "plan_user",
        "account_creation_timestamp_user",
    ]
)

# 特徴量とラベルの分離
features = cleaning_data.drop(columns=["rating", "rating_conv"])
labels = cleaning_data["rating_conv"]

# データセット準備
full_embedding_dims = {
    "gender_user": 2,
    "gender_target": 2,
    "location_user": 10,
    "location_target": 10,
    "age_range_user": 5,
    "age_range_target": 5,
    "height_range_user": 5,
    "height_range_target": 5,
    "body_type_user": 8,
    "body_type_target": 8,
    "personality_user": 10,
    "personality_target": 10,
    "appearance_user": 8,
    "appearance_target": 8,
    "job_user": 15,
    "job_target": 15,
    "blood_type_user": 4,
    "blood_type_target": 4,
    "car_user": 2,
    "car_target": 2,
    "interests_user": 20,
    "interests_target": 20,
    "salary_user": 10,
    "salary_target": 10,
}

user_embedding_dims = {
    "gender_user": 2,
    "location_user": 10,
    "age_range_user": 5,
    "height_range_user": 5,
    "body_type_user": 8,
    "personality_user": 10,
    "appearance_user": 8,
    "job_user": 15,
    "blood_type_user": 4,
    "car_user": 2,
    "interests_user": 20,
    "salary_user": 10,
}


# Preprocessorインスタンスの作成
preprocessor = Preprocessor(features, full_embedding_dims)
user_preprocessor = Preprocessor(user_data, user_embedding_dims)

# vocabulariesの確認
vocabularies = preprocessor.vocabularies
user_vocabularies = user_preprocessor.vocabularies

# feature_columnsの生成
feature_columns = preprocessor.create_feature_columns()
user_feature_columns = user_preprocessor.create_feature_columns()

# Embedding辞書の作成
embedding_dict = create_embedding_dict(feature_columns)
user_embedding_dict = create_embedding_dict(user_feature_columns)

# データの前処理
processed_data = preprocessor.process(features)
user_processed_data = user_preprocessor.process(user_data)

# --------------- #
# model の読み込み #
# --------------- #

# モデルの読み込み
loaded_model = tf.keras.models.load_model(f"{base_dir}/two_tower_model.keras")

# カスタムオブジェクトの登録
tf.keras.utils.get_custom_objects().update(
    {
        "SparseFeat": SparseFeat,
        "UserModel": UserModel,
        "ItemModel": ItemModel,
        "LightSE": LightSE,
        "DNN": DNN,
        "TwoTowerModel": TwoTowerModel,
    }
)

# モデルの確認
loaded_model.summary()

# item_modelを使用してターゲットデータの埋め込みを作成
item_inputs = {key: tf.constant(value) for key, value in processed_data.items() if key.endswith('_target')}
item_model = loaded_model.item_model
item_embeddings = item_model(item_inputs)

# ScaNNの設定
scann = tfrs.layers.factorized_top_k.ScaNN(num_reordering_candidates=1000)
scann.index(item_embeddings)

# 推薦関数の定義
def recommend(user_inputs):
    user_model = loaded_model.user_model
    user_embedding = user_model(user_inputs)
    scores, indices = scann(user_embedding)
    return scores, indices

# ユーザーデータをuser_modelに入力し、推薦アイテムを取得
scores, indices = recommend(user_processed_data)

print("User ID:", user_id)
print("推薦スコア:", scores)
print("推薦アイテムのインデックス:", indices)
