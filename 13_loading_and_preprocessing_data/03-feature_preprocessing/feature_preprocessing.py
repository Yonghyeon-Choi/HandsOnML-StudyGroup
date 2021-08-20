import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
assert tf.__version__ >= "2.0"
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_mean = scaler.mean_
X_std = scaler.scale_

HOUSING_PATH = os.path.join("datasets", "housing")


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
print()
print("# housing data keys #")
print(housing.keys())

print()
print("# housing_median_age #")
print(housing['housing_median_age'])

housing_median_age = tf.feature_column.numeric_column("housing_median_age")

age_mean, age_std = X_mean[1], X_std[1]
housing_median_age = tf.feature_column.numeric_column(
    "housing_median_age", normalizer_fn=lambda x: (x - age_mean) / age_std)

print("# housing_median_age 를 정규화한 텐서 #")
print(housing_median_age)

print()
print("# bucketized_age 텐서 #")
print("#  housing_median_age 텐서를 구간 분할해 범주화한 텐서 #")
print('# ~1.0 / -1.0~-0.5 / -0.5~0 / 0~0.5 / 0.5~1.0 / 1.0~ #')
bucketized_age = tf.feature_column.bucketized_column(
    housing_median_age, boundaries=[-1., -0.5, 0., 0.5, 1.])

print()
print("# median_income #")
print(housing['median_income'])
median_income = tf.feature_column.numeric_column("median_income")
print("# median_income 텐서 #")
print(median_income)

print()
print("# bucketized_income 텐서 #")
print("# median_income 텐서를 구간 분할해 범주화한 텐서 #")
print('# ~1.5 / 1.5~3.0 / 3.0~4.5 / 4.5~6.0 / 6.0~ #')
bucketized_income = tf.feature_column.bucketized_column(
    median_income, boundaries=[1.5, 3., 4.5, 6.])
print(bucketized_income)

print()
print("# ocean_proximity #")
print(housing['ocean_proximity'])
ocean_prox_vocab = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
ocean_proximity = tf.feature_column.categorical_column_with_vocabulary_list(
    "ocean_proximity", ocean_prox_vocab)
print("# ocean_proximity 텐서 #")
print("# ocean_proximity 를 사전형 텐서로 변환한 텐서 #")
print(ocean_proximity)

print()
print("# age_and_ocean_proximity 텐서 #")
print("# bucketized_age, ocean_proximity 텐서를 묶은 텐서 #")
print("# [bucketized_latitude, bucketized_longitude] #")
age_and_ocean_proximity = tf.feature_column.crossed_column(
    [bucketized_age, ocean_proximity], hash_bucket_size=100)

print()
print("# latitude, longitude 텐서 #")
latitude = tf.feature_column.numeric_column("latitude")
longitude = tf.feature_column.numeric_column("longitude")
print(latitude)
print(longitude)

print()
print("# bucketized_latitude, bucketized_longitude 텐서 #")
print("# latitude, longitude 를 범주화한 텐서 #")
print('# np.linspace(32., 42., 19) 32.0~42.0에서 19개의 구간으로 리스트 반환 #')
bucketized_latitude = tf.feature_column.bucketized_column(
    latitude, boundaries=list(np.linspace(32., 42., 19)))
bucketized_longitude = tf.feature_column.bucketized_column(
    longitude, boundaries=list(np.linspace(-125., -114., 19)))
print(bucketized_latitude)
print(bucketized_longitude)

print()
print("# location 텐서 #")
print("# latitude, longitude 를 범주화한 텐서를 묶은 텐서 #")
print("# [bucketized_latitude, bucketized_longitude] #")
location = tf.feature_column.crossed_column(
    [bucketized_latitude, bucketized_longitude], hash_bucket_size=1000)
print(location)

print()
print("# ocean_proximity_one_hot 텐서 #")
print("# ocean_proximity 텐서를 원-핫 인코딩한 텐서 #")
ocean_proximity_one_hot = tf.feature_column.indicator_column(ocean_proximity)
print(ocean_proximity_one_hot)

print()
print("# ocean_proximity 텐서를 임베딩한 텐서 #")
ocean_proximity_embed = tf.feature_column.embedding_column(ocean_proximity,
                                                           dimension=2)
print(ocean_proximity_embed)

ocean_prox_vocab = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
indices = tf.range(len(ocean_prox_vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(ocean_prox_vocab, indices)
num_oov_buckets = 1
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)

categories = tf.constant(['<1H OCEAN', 'SUWON', 'NEAR BAY'])
cat_indices = table.lookup(categories)
print("# 원-핫 벡터를 이용한 특성 인코딩 #")
print(cat_indices)
cat_one_hot = tf.one_hot(cat_indices, depth=len(ocean_prox_vocab) + num_oov_buckets)
print(cat_one_hot)

print()
print("# 임베딩을 이용한 특성 인코딩 #")
embedding_dim = 2
embed_init = tf.random.uniform([len(ocean_prox_vocab) + num_oov_buckets, embedding_dim])
embedding_matrix = tf.Variable(embed_init)
print(embedding_matrix)

categories = tf.constant(['<1H OCEAN', 'SUWON', 'NEAR BAY'])
cat_indices = table.lookup(categories)
print(tf.nn.embedding_lookup(embedding_matrix, cat_indices))

embedding = keras.layers.Embedding(input_dim=len(ocean_prox_vocab) + num_oov_buckets,
                                   output_dim=embedding_dim)
print(embedding(cat_indices))

print()
print("# 케라스 모델 만들기 #")
regular_inputs = keras.layers.Input(shape=[8])
categories = keras.layers.Input(shape=[], dtype=tf.string)
cat_indices = keras.layers.Lambda(lambda cats: table.lookup(cats))(categories)
cat_embed = keras.layers.Embedding(input_dim=6, output_dim=2)(cat_indices)
encoded_inputs = keras.layers.concatenate([regular_inputs, cat_embed])
outputs = keras.layers.Dense(1)(encoded_inputs)
model = keras.models.Model(inputs=[regular_inputs, categories],
                           outputs=[outputs])




