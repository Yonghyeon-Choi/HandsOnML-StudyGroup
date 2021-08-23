import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
assert tf.__version__ >= "2.0"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


'''
# Example 프로토콜 버퍼 정의 #

proto
syntax = "proto3";

message BytesList { repeated bytes value = 1; }
message FloatList { repeated float value = 1 [packed = true]; }
message Int64List { repeated int64 value = 1 [packed = true]; }
message Feature {
    oneof kind {
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};
message Features { map<string, Feature> feature = 1; };
message Example { Features features = 1; };
'''

BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example

# from tensorflow.train import BytesList, FloatList, Int64List  # 위 33 ~ 38줄과 동일
# from tensorflow.train import Feature, Features, Example

print()
print("# [name : Alice, id : 123, emails : a@b.com, c@d.com] #")
print("# Example 프로토콜 버퍼를 사용해 특성을 가진 person_example 객체 생성 #")
person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com", b"c@d.com"]))
        }))
print(person_example)

print("# person_example 객체를 직열화하여 파일 출력 #")
with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
    f.write(person_example.SerializeToString())

print()
print("# 파일 입력을 받기위한 person_example 객체를 초기화할 placeholder 구성 #")
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}

print()
print("# person_example 파일 입력을 받아 파싱하여 저장 #")
parsed_example = ""
for serialized_example in tf.data.TFRecordDataset(["my_contacts.tfrecord"]):
    parsed_example = tf.io.parse_single_example(serialized_example,
                                                feature_description)
print()
print(parsed_example)
print()
print(parsed_example["name"])   # 파싱된 특성 살펴보기
print(parsed_example["id"])
print(parsed_example["emails"].values)
# print(parsed_example["emails"].values[0])
# print(tf.sparse.to_dense(parsed_example["emails"], default_value=b""))
print("=====================================================================")

'''
# SequenceExample 프로토콜 버퍼 정의 #

proto
syntax = "proto3";

message FeatureList { repeated Feature feature = 1; };
message FeatureLists { map<string, FeatureList> feature_list = 1; };
message SequenceExample {
  Features context = 1;
  FeatureLists feature_lists = 2;
};
'''

FeatureList = tf.train.FeatureList
FeatureLists = tf.train.FeatureLists
SequenceExample = tf.train.SequenceExample

# from tensorflow.train import FeatureList, FeatureLists, SequenceExample   # 위 96 ~ 98줄과 동일

print()
print("# [author_id : 123, title : A, desert, place, ., pub_date : 1623, 12, 15] #")
print("# Example 프로토콜 버퍼를 사용해 특성을 가진 context 객체 생성 #")
context = Features(feature={
    "author_id": Feature(int64_list=Int64List(value=[123])),
    "title": Feature(bytes_list=BytesList(value=[b"A", b"desert", b"place", b"."])),
    "pub_date": Feature(int64_list=Int64List(value=[1623, 12, 25]))
})

print()
print("# 특성 리스트에 사용할 리스트 #")
content = [["When", "shall", "we", "three", "meet", "again", "?"],
           ["In", "thunder", ",", "lightning", ",", "or", "in", "rain", "?"]]
comments = [["When", "the", "hurlyburly", "'s", "done", "."],
            ["When", "the", "battle", "'s", "lost", "and", "won", "."]]

print()
print("# Example 프로토콜 버퍼를 사용해 특성 리스트를 만드는 함수 #")
def words_to_feature(words):
    return Feature(bytes_list=BytesList(value=[word.encode("utf-8")
                                               for word in words]))


content_features = [words_to_feature(sentence) for sentence in content]
comments_features = [words_to_feature(comment) for comment in comments]

print()
print("# SequenceExample 프로토콜 버퍼를 사용해 \"특성, 특성 리스트, 특성 리스트\"을 가진 sequence_example 객체 생성 #")
sequence_example = SequenceExample(
    context=context,
    feature_lists=FeatureLists(feature_list={
        "content": FeatureList(feature=content_features),
        "comments": FeatureList(feature=comments_features)
    }))
print(sequence_example)

print("# sequence_example 객체를 직열화 #")
serialized_sequence_example = sequence_example.SerializeToString()
print(serialized_sequence_example)

print()
print("# 직렬화한 sequence_example 객체를 초기화할 placeholder 구성 #")
context_feature_descriptions = {
    "author_id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "title": tf.io.VarLenFeature(tf.string),
    "pub_date": tf.io.FixedLenFeature([3], tf.int64, default_value=[0, 0, 0]),
}
sequence_feature_descriptions = {
    "content": tf.io.VarLenFeature(tf.string),
    "comments": tf.io.VarLenFeature(tf.string),
}

print()
print("# 직렬화한 sequence_example 객체를 파싱 #")
parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
    serialized_sequence_example,
    context_feature_descriptions, sequence_feature_descriptions)
print()
print(parsed_context)
print(parsed_context["title"].values)   # 파싱된 특성 살펴보기
print()
print(parsed_feature_lists['content'].values)
print(parsed_feature_lists['content'])
print(tf.RaggedTensor.from_sparse(parsed_feature_lists["content"]))
# Ragged Array : 비정형 배열, 서로 다른 길이의 일차원 배열들을 묶어 놓은 2차원 배열 (딱보기에 계단형으로 생긴 배열)
# tf.RaggedTensor API 에 다양한 비정형 텐서를 다루는 함수가 있음
# https://www.tensorflow.org/api_docs/python/tf/RaggedTensor
# Sparse Matrix : 희소 행렬
# tf.RaggedTensor.from_sparse 는 희소텐서를 비정형텐서로 변환하는 함수

