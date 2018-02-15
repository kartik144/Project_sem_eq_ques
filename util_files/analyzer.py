import nltk
import matplotlib.pyplot as plt
import math
from nltk.corpus import stopwords
from operator import itemgetter

def get_data(input_file):
	dataset=list([])
	tokens=set([])
	f=open(input_file,"r")
	s=f.readline()
	while s!="":
		s=s.replace("/"," / ") 
		dataset.append(s)
		tokens=tokens.union(set([w.lower() for w in nltk.word_tokenize(s)]))
		s=f.readline()
	
	return (tokens,dataset)

def build_dictionary(dataset,tokens):
	word_dict={}
	for w in tokens:
		word_dict[w]=(0,0)
	
	for d in dataset:
		words=set([w.lower() for w in nltk.word_tokenize(d)])
		for w in nltk.word_tokenize(d):
			w=w.lower()
			count,docs=word_dict[w]
			count=count+1
			word_dict[w]=(count,docs) 
		
		for w in words:
			count,docs=word_dict[w]
			docs=docs+1
			word_dict[w]=(count,docs) 
	
	dictionary={}
	
	for k in word_dict.keys():
		count,docs=word_dict[k]
		val=math.log1p(count)*math.log1p(len(dataset)/docs)
		
		dictionary[k]=val
	
	return dictionary

print("Enter path of file: ",end="")
input_file=input()
	
tokens,dataset = get_data(input_file)
word_dict = build_dictionary(dataset,tokens)
word_freq=[]
X=[]
Y=[]
X_stopwords=[]
Y_stopwords=[]
for w in word_dict.keys():
	word_freq.append((w,word_dict[w]))
word_freq=sorted(word_freq,key=itemgetter(1))
sorted_X=[x[0] for x in word_freq]
sorted_Y=[x[1] for x in word_freq]
stopWords = set(stopwords.words('english'))

for x in range(0,len(sorted_X)):
	if sorted_X[x] in stopWords:
		X_stopwords.append(x)
		Y_stopwords.append(sorted_Y[x])
	else:
		X.append(x)
		Y.append(sorted_Y[x])

plt.title("Tokens and their TF-IDF")
plt.xlabel("Words")
plt.ylabel("TF-IDF")
plt.plot(X_stopwords,Y_stopwords,'-r',label = "Stop Words")
plt.plot(X,Y,label = "Other Words")
plt.legend(loc='upper left')

plt.show()

