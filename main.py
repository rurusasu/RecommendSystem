import tensorflow as tf
import pickle
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import datetime

from src.feat import SparseFeat
from src.layers import DNN, LightSE
from src.models import ItemModel, TwoTowerModel, UserModel
from src.processor import Preprocessor
from src.utils import create_embedding_dict

base_dir = "/home/user/core/data"

# データ読み込み例
# data = pd.read_excel(f"{base_dir}/sample_merged_full.xlsx")
#data = pd.read_excel(f"{base_dir}/merged_full_rating_conv.xlsx")

#with open(f"{base_dir}/sample_merged_full_rating_conv.pkl", "rb") as f:
with open(f"{base_dir}/merged_full_rating_conv.pkl", "rb") as f:
    data = pickle.load(f)

# ラベルに不正な値が含まれている可能性があるので、削除
cleaning_data = data[data["rating"].isin([0, 1, 2])]
cleaning_data = cleaning_data.dropna()

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

# 特徴量とラベルの分離
features = cleaning_data.drop(columns=["rating", "rating_conv"])
labels = cleaning_data["rating_conv"]

# データセット準備
embedding_dims = {
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

# Preprocessorインスタンスの作成
preprocessor = Preprocessor(features, embedding_dims)

# vocabulariesの確認
vocabularies = preprocessor.vocabularies

# feature_columnsの生成
feature_columns = preprocessor.create_feature_columns()

# Embedding辞書の作成
embedding_dict = create_embedding_dict(feature_columns)

# データの前処理
processed_data = preprocessor.process(features)

dataset = tf.data.Dataset.from_tensor_slices((dict(processed_data), labels))

# データセットを訓練データと検証データに分割
dataset_size = len(cleaning_data)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

# シャッフルしてから分割
dataset = dataset.shuffle(buffer_size=dataset_size)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# モデルの訓練に適したようにデータセットをバッチ化
train_dataset = train_dataset.batch(128).repeat()
val_dataset = val_dataset.batch(128).repeat()

# Two Towerモデルの定義
layer_sizes = [128, 64, 32]
field_size = 12
embedding_size = 32
use_bn = False
use_ln = True
activation = "silu"

epochs = 300
verbose = False

# modelの読み出し
two_tower_model = TwoTowerModel(
    layer_sizes=layer_sizes,
    field_size=field_size,
    embedding_size=embedding_size,
    embedding_dict=embedding_dict,
    feature_columns=feature_columns,
    activation=activation,
    use_bn=use_bn,
    use_ln=use_ln,
    verbose=verbose,
)

# オプティマイザー
optimizer = Adam(learning_rate=0.001)

# メトリック
train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(
    name="train_accuracy"
)
val_loss_metric = tf.keras.metrics.Mean(name="val_loss")
val_accuracy_metric = tf.keras.metrics.CategoricalAccuracy(name="val_accuracy")

# TensorBoardのセットアップ
log_dir = f"{base_dir}/tensorborad_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 訓練ステップ
@tf.function
def train_step(inputs, labels, verbose: bool = False):
    with tf.GradientTape() as tape:
        similarity = two_tower_model(inputs)

        # ラベルの範囲を確認
        if verbose:
            tf.print(
                "ラベルの範囲:", tf.reduce_min(labels), tf.reduce_max(labels)
            )

        # コサイン類似度に対する適切な損失関数を適用
        loss = tf.reduce_mean(
            tf.square(similarity - tf.cast(labels, tf.float32))
        )

    # 勾配の計算とオプティマイザーの適用
    gradients = tape.gradient(loss, two_tower_model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, two_tower_model.trainable_variables, strict=False)
    )

    train_loss_metric(loss)
    train_accuracy_metric(labels, similarity)


# 評価ステップ
@tf.function
def val_step(inputs, labels):
    similarity = two_tower_model(inputs)
    loss = tf.reduce_mean(tf.square(similarity - tf.cast(labels, tf.float32)))

    val_loss_metric(loss)
    val_accuracy_metric(labels, similarity)


train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

steps_per_epoch = train_size // 128
validation_steps = val_size // 128

for epoch in range(epochs):

    # 訓練データセットの反復
    for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=False):
        batch = next(iter(train_dataset))
        inputs, labels = batch
        train_step(inputs, labels, verbose)

    # 訓練の損失と精度を記録
    train_loss = train_loss_metric.result().numpy()
    train_accuracy = train_accuracy_metric.result().numpy()
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 検証データセットの評価
    for _ in tqdm(range(validation_steps), desc=f"Epoch {epoch + 1}/{epochs} - Validation", leave=False):
        val_batch = next(iter(val_dataset))
        inputs, labels = val_batch
        val_step(inputs, labels)

    # 検証の損失と精度を記録
    val_loss = val_loss_metric.result().numpy()
    val_accuracy = val_accuracy_metric.result().numpy()
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # tqdmの更新
    tqdm.write(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # if (
    #    epoch >= 20 and (epoch + 1) % 10 == 0
    # ):  # 最初の20エポックはスキップし、10エポックごとにグラフを更新
    # プロット
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label="Training Loss")
    # plt.plot(val_losses, label="Validation Loss")
    # plt.legend(loc="upper right")
    # plt.title("Training and Validation Loss")

    # plt.subplot(1, 2, 2)
    # plt.plot(train_accuracies, label="Training Accuracy")
    # plt.plot(val_accuracies, label="Validation Accuracy")
    # plt.legend(loc="lower right")
    # plt.title("Training and Validation Accuracy")

    # plt.show()

    # TensorBoardのログ
    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.scalar('Train Loss', train_loss, step=epoch)
        tf.summary.scalar('Train Accuracy', train_accuracy, step=epoch)
        tf.summary.scalar('Validation Loss', val_loss, step=epoch)
        tf.summary.scalar('Validation Accuracy', val_accuracy, step=epoch)

    # メトリックのリセット
    train_loss_metric.reset_state()
    train_accuracy_metric.reset_state()
    val_loss_metric.reset_state()
    val_accuracy_metric.reset_state()

# モデルの確認
two_tower_model.summary()

# モデルの保存
two_tower_model.save(f"{base_dir}/two_tower_model.keras")

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

# モデルの読み込み
loaded_model = tf.keras.models.load_model(f"{base_dir}/two_tower_model.keras")

# モデルの確認
loaded_model.summary()
