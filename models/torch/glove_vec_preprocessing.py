import numpy as np
import bcolz
import pickle

words = []
idx = 0
word2idx = {}
dim = 200
glove_dir = "../../glove.6B"
glove_path = glove_dir+"/glove.6B."+str(dim)+"d.txt"
vectors = bcolz.carray(np.zeros(1), rootdir=glove_dir+'/6B.'+str(dim)+'.dat', mode='w')

with open(glove_path, 'rb') as f:
    for l in f:
        # print(l)
        line = l.decode('utf-8').split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)


vectors = bcolz.carray(vectors[1:].reshape((-1, dim)), rootdir=glove_dir+'/6B.'+str(dim)+'.dat', mode='w')
vectors.flush()
pickle.dump(words, open(glove_dir+'/6B.'+str(dim)+'_words.pkl', 'wb'))
pickle.dump(word2idx, open(glove_dir+'/6B.'+str(dim)+'_idx.pkl', 'wb'))
