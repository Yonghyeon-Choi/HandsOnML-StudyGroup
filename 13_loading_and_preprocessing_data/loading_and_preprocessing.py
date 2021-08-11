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

np.random.seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "data"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
#############################################################################


print()
print("# dataSet = tf.data.Dataset.range(10) #")
X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
# dataset = tf.data.Dataset.range(10) # 위 두줄과 동일
print(dataset)

print()
print("# item in dataset #")
for item in dataset:
    print(item)
tmp = dataset

print()
print("# repeat(3).batch(7) | 3번 반복, 7 배치 사이즈 #")
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

print()
print("# repeat(3).batch(7, drop_remainder=True) | 나머지 버리기 #")
dataset_drop = tmp.repeat(3).batch(7, drop_remainder=True)
for item in dataset_drop:
    print(item)

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

print()
print("# filter(lambda x: x < 10) | 10 미만#")
dataset = dataset.filter(lambda x: x < 10)
for item in dataset:
    print(item)

print()
print("# take(3) | 3개의 아이템만 #")
for item in dataset.take(3):
    print(item)

print()
print("# show 3 items #")
tf.random.set_seed(42)

dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=3, seed=42).batch(7)
for item in dataset:
    print(item)