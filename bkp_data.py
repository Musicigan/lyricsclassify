import numpy as np
import pandas as pd
from nltk import sent_tokenize, word_tokenize
import ast
import json

df = pd.read_csv('lyrics.csv',low_memory=False)

genre_dict={}

data=[]
newdata={}
count =0
for index,l, g in zip(df.index.values,df.lyrics.values,df.genre.values):
	count+=1
	if count>10:
		break
	if type(l) is str and (g!='Not Available' or g!='Other'):
		newdata={}
		newdata['index']=index
		newdata['genre']=g	
		l=l.decode('utf-8')
        newdata['lyrics']=l
        data.append(newdata)

# data is a list of dictionaries
# data stores keys alphabetically  so 0->genre 1-> index 2-> lyrics

# print np.shape(data)
# print data[0].values()[0] # prints lyrics

#******************************making the embedding matrix***********************************************

top = 30000
embedding_matrix = np.zeros((top+2,100))
word_dict={}
for i,ele in enumerate(data):
	line = ele.values()[2]
	inputs = line.split()
	word=inputs[0]
	vec = np.array([float() for x in inputs[1:]])
	embedding_matrix[i,:] = vec[0:100]
	word_dict[word] = i


np.save('embeddingMatrix',embedding_matrix)
print("Finished embeddings")

def link_word(word):
    if word in word_dict:
        return word_dict[word]
    else:
        return word_dict['<unk>']


outfile = 'indexed_data.json'
open(outfile,'w').close()
# with open('tokenized_data.json','r') as f:
for i, ele in enumerate(data):
    # datapoint = json.loads(line)
    lyr = ele.values()[2]
    sents = sent_tokenize(lyr)
    words = [word_tokenize(sent) for sent in sents]
    idx_lines = []
    for l in lyr:
        idxs = [link_word(word) for word in l]
        idx_lines.append(idxs)

    new_dict = {}
    new_dict['genre'] = lyr = ele.values()[0]
    new_dict['idxs'] = idx_lines
    with open(outfile,'a') as g:
        g.write(json.dumps(new_dict))
        g.write('\n')

    if (i % 50000 == 0):
        print("Completed:", i)

