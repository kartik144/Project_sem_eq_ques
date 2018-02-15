import gensim

gmodel=gensim.models.KeyedVectors.load_word2vec_format("GloVe/glove.6B/glove_transfer_word2vec.6B.100d.txt",binary=False)
#ms=gmodel.most_similar('frog')
#print(ms)

gmodel.wv.similarity('woman', 'man')
