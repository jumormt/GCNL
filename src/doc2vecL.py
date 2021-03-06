# https://blog.csdn.net/john_xyz/article/details/79208564
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# for i,doc in enumerate(common_texts):
#     print(doc)
#     print(i)
documents = [TaggedDocument(doc, [str(i)]) for i, doc in enumerate(common_texts)]
documents1 = documents[:3]
documents2 = documents[3:]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# todo:增量学习 python会意外退出 未知原因
# model.build_vocab(documents=documents2, update=True)
#
# model.train(documents2, total_examples=model.corpus_count, epochs=model.iter)
# 与标签‘0’最相似的
model.docvecs.most_similar("0")
# 进行相关性比较
print(model.docvecs.similarity('0','1'))
# 输出标签为‘10’句子的向量
print(model.docvecs['10'])
# 也可以推断一个句向量(未出现在语料中)
words = u"여기 나오는 팀 다 가슴"
print(model.infer_vector(words.split()))
# 也可以输出词向量
print(model[u'human'])

model.infer_vector(['human'])
# print(model.docvecs.distance())
print(model.corpus_count)

print("end")
