import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# np.random.seed(42)
#
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)
#
# # 그림을 저장할 위치
# PROJECT_ROOT_DIR = "."
# CHAPTER_ID = "data"
# IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
# os.makedirs(IMAGES_PATH, exist_ok=True)
#
#
# def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#     path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
#     print("그림 저장:", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format=fig_extension, dpi=resolution)
#############################################################################

X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
# dataset = tf.data.Dataset.range(10) # 위 두줄과 동일
print()
print("# dataSet = tf.data.Dataset.range(10) #")
print(dataset)

print()
print("# item in dataset #")
for item in dataset:
    print(item)
tmp = dataset

# repeat(): 원본 데이터셋의 아이템을 N차례 반복하는 새로운 데이터셋을 반환 (복사하는 것은 아님)
# batch() : 아이템을 N개의 그룹으로 묶는다
print()
print("# repeat(3).batch(7) | 3번 반복, 7 배치 사이즈 #")
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

# batch(drop_remainder=True): 마지막에 N보다 부족한 길이의 배치는 버림 (=모든 배치의 크기가 동일)
print()
print("# repeat(3).batch(7, drop_remainder=True) | 나머지 버리기 #")
dataset_drop = tmp.repeat(3).batch(7, drop_remainder=True)
for item in dataset_drop:
    print(item)

# 데이터에 원하는 전처리 작업에도 적용 (이미지 크기 변환, 회전계산)
# map(num_parallel_calls) 를 하면 여러개의 스레드로 나누어서 속도를 높여 처리 가능
print()
print("# map(lambda x: x * 2) | 간단한 전처리(*2) #")
dataset = dataset.map(lambda x: x * 2)
for item in dataset:
    print(item)


print()
print("# unbatch #")
# dataset = dataset.apply(tf.data.experimental.unbatch())
dataset = dataset.unbatch()
for item in dataset:
    print(item)

# filter 할때
print()
print("# filter(lambda x: x < 10) | 10 미만 #")
dataset = dataset.filter(lambda x: x < 10)
for item in dataset:
    print(item)

# 데이터셋에 있는 몇개의 아이템만 보고싶을때
print()
print("# take(3) | 3개의 아이템만 #")
for item in dataset.take(3):
    print(item)

# 경사 하강법은 훈련 세트에 있는 샘플이 독립적이고 동일한 분포일때 최고의 성능을 발휘 => shuffle이 필요한 이유
# 1. 원본 데이터셋의 처음 아이템을 buffer_size 개수만큼 추출 버퍼에 채움
# 2. 새로운 아이템이 요청되면 이 버퍼에서 랜덤하게 하나를 꺼내 반환
# 3. 원본 데이터셋에서 새로운 아이템을 추출하여 비워진 버퍼를 채움
# 4. 원본 데이터셋의 모든 아이템이 사용될 때까지 반복
# 5. 버퍼가 비워질 때까지 계속하여 랜덤하게 아이템을 반환
# 버퍼의 크기를 충분히 크게 하는 것이 중요 => 셔플링 효과를 올리기 위해, 메모리의 크기 넘지 X, 데이터의 크기 넘지 X
# 완벽한 셔플링을 위해서는 버퍼크기가 데이터셋의 크기와 동일
print()
print("# shuffle(buffer_size=30, seed=42).batch(10) | 섞기 #")
tf.random.set_seed(42)
dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=30, seed=42).batch(10)
for item in dataset:
    print(item)

#############################################################################
# 데이터 적재 #
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_mean = scaler.mean_
X_std = scaler.scale_


def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = os.path.join("datasets", "housing")
    os.makedirs(housing_dir, exist_ok=True)
    path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

    filepaths = []
    m = len(data)
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths


train_data = np.c_[X_train, y_train]
valid_data = np.c_[X_valid, y_valid]
test_data = np.c_[X_test, y_test]
header_cols = housing.feature_names + ["MedianHouseValue"]
header = ",".join(header_cols)

train_filepaths = save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
valid_filepaths = save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
test_filepaths = save_to_multiple_csv_files(test_data, "test", header, n_parts=10)

import pandas as pd

print()
print("# 데이터 적재 | dataset head #")
print(pd.read_csv(train_filepaths[0]).head())

print()
print("# 데이터 적재 | shuffle read #")
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)

for filepath in filepath_dataset:
    print(filepath)

# skip(1): header
# interleave(): filepath_dataset에 있는 다섯개의 파일 경로에서 데이터를 읽는 데이터셋을 생성,
#               TextLineDataset 5개를 순회하면서 한줄씩 읽음
# 파일 길이가 동일할때 interleave를 사용하는게 좋음 (각파일에서 한줄씩 읽음)
# num_parallel_calls 매개변수에 원하는 스레드 개수를 지정
# tf.data.experimental.AUTOTUNE: 을 지정하면 텐서플로가 가용한 CPU를 기반으로 동적으로 적절한 스레드 개수를 선택할 수 있음
# cycle_length: 동시에 처리할 입력 개수를 지정
n_readers = 5
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers)

print()
print("# 데이터 적재 | interleave #")
print("# train 13, 14, 09, 17, 01의 index 0 샘플 #")
for line in dataset.take(5):
    print(line.numpy())

#############################################################################
# 전처리 #
n_inputs = 8    # X_train.shape[-1]


@tf.autograph.experimental.do_not_convert
@tf.function
def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y


print()
print("# 전처리 | 특성 스케일 조정 #")
print("# (예시 : [4.2083, 44.0, 5.3232, 0.9171, 846.0, 2.3370, 37.47, -122.2, 2.782]) #")
print(preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782'))

#############################################################################
# 데이터 적재 + 전처리 #


def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)


tf.random.set_seed(42)

print()
print("# 데이터 적재와 전처리 합치기 #")
print("# 캘리포니아 주택 데이터 적재 + 스케일 조정 전처리 #")
print()
print("# 총 11609개 샘플이 있어, 상단 1개 샘플만 표기 #")
train_set = csv_reader_dataset(train_filepaths, batch_size=1)
for X_batch, y_batch in train_set.take(1):
    print("X =", X_batch)
    print("y =", y_batch)
    print()

train_set = csv_reader_dataset(train_filepaths, repeat=None)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1),
])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
batch_size = 32
model.fit(train_set, steps_per_epoch=len(X_train) // batch_size, epochs=10,
          validation_data=valid_set)
model.evaluate(test_set, steps=len(X_test) // batch_size)

new_set = test_set.map(lambda X, y: X)
X_new = X_test
print()
print(model.predict(new_set, steps=len(X_new) // batch_size))

#################################################################################
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

optimizer = keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = keras.losses.mean_squared_error


@tf.function
def train(model, n_epochs, batch_size=32,
          n_readers=5, n_read_threads=5, shuffle_buffer_size=10000, n_parse_threads=5):
    train_set = csv_reader_dataset(train_filepaths, repeat=n_epochs, n_readers=n_readers,
                       n_read_threads=n_read_threads, shuffle_buffer_size=shuffle_buffer_size,
                       n_parse_threads=n_parse_threads, batch_size=batch_size)
    for X_batch, y_batch in train_set:
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


train(model, 5)

print()
print("# 전체 훈련 반복 수행 텐서플로 함수 #")
print(model.predict(new_set, steps=len(X_new) // batch_size))
