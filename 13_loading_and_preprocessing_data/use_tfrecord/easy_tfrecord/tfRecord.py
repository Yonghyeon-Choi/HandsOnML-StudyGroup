import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = ''


with tf.io.TFRecordWriter("my_data.tfrecord") as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")

filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
print()
print("# 간단한 문자열 TFRecord #")
print("with tf.io.TFRecordWriter(\"my_data.tfrecord\") as f:")
print("\tf.write(b\"This is the first record\")")
print("\tf.write(b\"And this is the second record\")")
print()
for item in dataset:
    print(item)

print()
print("# 간단한 반복 TFRecord #")
filepaths = ["my_test_{}.tfrecord".format(i) for i in range(5)]
for i, filepath in enumerate(filepaths):
    with tf.io.TFRecordWriter(filepath) as f:
        for j in range(3):
            f.write("File {} record {}".format(i, j).encode("utf-8"))

dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)

print()
print("# 압축 TFRecord #")
print("# 옵션 options = tf.io.TFRecordOptions(compression_type=\"GZIP\") #")
print("# 압축 with tf.io.TFRecordWriter(\"my_compressed.tfrecord\", options) as f: #")
print("# 해제 tf.data.TFRecordDataset([\"my_compressed.tfrecord\"], compression_type=\"GZIP\") #")
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")

dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"], compression_type="GZIP")
for item in dataset:
    print(item)

print()
print("# 텐서플로 프로토콜 버퍼 #")
print("# Person 객체 생성 / 식별자 name : 1, id : 2, email : 3 #")
print("# person = Person(name=\"Al\", id=123, email=[\"a@b.com\"]) #")
from person_pb2 import Person

person = Person(name="Al", id=123, email=["a@b.com"])  # Person 생성
print(person)

print("# person 객체 수정 #")
person.name = "Alice"           # 이름 수정
person.email.append("c@d.com")  # 이메일 추가
print(person)

print("# person 객체 직렬화 #")
print("# s = person.SerializeToString() #")
s = person.SerializeToString()  # 바이트 문자열로 직렬화
print(s)

print()
print("# person 객체 직렬화한 문자열을 파싱하여 복원 #")
print("# person2.ParseFromString(s) #")
person2 = Person()              # 새로운 Person 생성
print(person2.ParseFromString(s))      # 바이트 문자열 파싱 (27 바이트)
print(person2)

