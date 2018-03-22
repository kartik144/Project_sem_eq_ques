import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.autograd import Variable
import pickle
from time import time
import os
import Tree
import nltk
import matplotlib.pyplot as plt

def getEmbeddings(word_vec_file="GloVe/glove.6B/glove.6B.50d.txt",embedding_dim=50):
	if(('word_vectors'+str(embedding_dim)+'.pt' in os.listdir(os.getcwd()+"/Pickles")) and ('words2idx'+str(embedding_dim)+'.pt' in os.listdir(os.getcwd()+"/Pickles"))):
		print("Retrieving pickled embeddings")
		t0=time()
		out_file1=open('Pickles/word_vectors'+str(embedding_dim)+'.pt','rb')
		out_file2=open('Pickles/words2idx'+str(embedding_dim)+'.pt','rb')
		embed=pickle.load(out_file1)
		words2idx=pickle.load(out_file2)
		print("Pickled GloVe loaded in {0:0.4f} seconds".format(time()-t0))
		return (embed,words2idx)
	
	print("Loading pre-trained GloVe vectors...")
	t0=time()
	words2idx={}
	word2vec={}
	matrix=[]
	f=open(word_vec_file,"r")
	i=0
	sen=f.readline()
	while(sen!=""):
		t=nltk.word_tokenize(sen)
		vecx=t[(-1*embedding_dim):]
		sent=t[:(-1*embedding_dim)]
		vec=[]
		sen=""
		for j in range(0,len(vecx)):
			vec.append(float(vecx[j]))
		for j in range(0,len(sent)):
			sen=sen+sent[j]
		
		words2idx[sen]=i
		matrix.append(vec)
		i=i+1
		sen=f.readline()
	
	embed = nn.Embedding(i, embedding_dim)
	embed.weight.data.copy_(torch.from_numpy(np.asmatrix(matrix)))
	embed.weight.requires_grad = False
	out_file1=open('Pickles/'+'word_vectors'+str(embedding_dim)+'.pt','wb')
	out_file2=open('Pickles/'+'words2idx'+str(embedding_dim)+'.pt','wb')
	pickle.dump(embed,out_file1)
	pickle.dump(words2idx,out_file2)
	#print(embed(Variable(torch.LongTensor([words2idx['the']]))))
	print(embed)
	print("GloVe loaded and pickled in %f seconds",str(time()-t0))
	return (embed,words2idx)

class TreeRNN(nn.Module):
	def __init__(self,labels,embedding_dim=50):
		super(TreeRNN, self).__init__()
		self.Wv=nn.Linear(embedding_dim,embedding_dim)
		self.W={}
		for l in labels:
			self.W[l]=nn.Linear(embedding_dim,embedding_dim)
		self.Dv=nn.Linear(embedding_dim,embedding_dim)
		self.D={}
		for l in labels:
			self.D[l]=nn.Linear(embedding_dim,embedding_dim)
		
	def synthesis(self,node,embedding_dim=50):
		if (node.children == []):
			name=""
			for i in range(0,len(node.data.split('-'))-1):
				name=name+"-"+node.data.split('-')[i]
			name=name[1:]
			
			### Hack for words not in vocabulary, needs tuning!! ###
			try:
				node.X=embed(torch.LongTensor([words2idx[name.lower()]]))
			except:
				missing_words.append(name.lower())
				node.X=Variable(torch.rand(1,embedding_dim))
			###########################################################
			node.H=F.tanh(self.Wv(node.X))
			#print(node.data)
			#print(node.H)
			return
		else:
			for c in node.children:
				self.synthesis(c,embedding_dim)
			s=Variable(torch.zeros(1,embedding_dim))
			name=""
			for i in range(0,len(node.data.split('-'))-1):
				name=name+"-"+node.data.split('-')[i]
			name=name[1:]
			### Hack for words not in vocabulary, needs tuning!! ###
			try:
				node.X=embed(torch.LongTensor([words2idx[name.lower()]]))
			except:
				missing_words.append(name.lower())
				node.X=Variable(torch.rand(1,embedding_dim))
			###########################################################
			for c in node.children:
				k=self.W[c.type](c.H)
				s=s+k
			node.H=F.tanh(self.Wv(node.X)+s)
			#print(node.data)
			#print(node.H)
			return
	def reconstruct(self,node,embedding_dim=50):
		if (node.parent != None):
			node.U=F.tanh(self.D[node.type](node.parent.U))########CHANGE
			node.R=F.tanh(self.Dv(node.U))########CHANGE
		if (node.children == []):
			return
		else:
			for c in node.children:
				self.reconstruct(c,embedding_dim)
	
	def getError(self,node,embedding_dim=50):
		if (node.children == []):
			return (((node.R - node.X)*(node.R - node.X))/embedding_dim)
		else:
			acc=Variable(torch.zeros(1,embedding_dim))
			for c in node.children:	
				acc=acc+self.getError(c,embedding_dim)
			if (node.parent != None):
				return ((((node.R - node.X)*(node.R - node.X))/embedding_dim)+acc)
			else:
				return (acc)
			
	def TreeRNN_MSE(self,node,embedding_dim):
		err=self.getError(node,embedding_dim)
		return err.sum()
		
	def forward(self,node,embedding_dim):
		self.synthesis(node,embedding_dim)
		node.U=node.H########CHANGE
		self.reconstruct(node,embedding_dim)

def generate_dataset(input_file="Input Files/NTU_ques_dependencies_parsed.txt"):

	print("Generating dataset...")
	t0=time()
	f=open(input_file,"r")
	sen=f.readline()
	dataset=[]
	i=0
	rejected=0
	while sen!="":
		cycle=False
		nodes={}
		tree_data=[]
		while (sen != "\n"):
			tree_data.append(sen)
			sen=f.readline()
		
		sen=f.readline()
		for s in tree_data:
			ty=(s.split('(')[0]).replace(":","_")
			from_node=(s.split('(')[1]).split(',')[0]
			to_node=((s.split('(')[1]).split(',')[1])[1:-2]
		
			if from_node not in nodes.keys():
				nodes[from_node]=[]
			nodes[from_node].append((to_node,ty))
		#Tree.print_parse_tree(nodes)
		
		for k in nodes.keys():
			for a,ty in nodes[k]:
				if a==k:
					rejected+=1
					cycle=True
		if (cycle==False):
			root=None
			root=Tree.TreeNode('ROOT-0')
			#print("Generating Tree Number : ",i)
			i+=1
			root.generate_parse_Tree(nodes)
			dataset.append(root)
	
	print(str(rejected)+" sentences rejected due to bad parsing!!")
	print(str(i)+" sentences appended to the dataset !!!")
	print("Dataset generated in {0:0.4f} seconds...".format(time()-t0))
	return dataset
	
def train_model(total_epochs,train_set,word_vec_file="GloVe/glove.6B/glove.6B.50d.txt",embedding_dim=50,lr=0.001):
	
	model=TreeRNN(labels,embedding_dim)
	
	#loss_function = model.TreeRNN_MSE()
	
	optimizer = optim.SGD(model.parameters(), lr)
	losses=[]
	epochs=[]
	total_time=time()
	for epoch in range(total_epochs):
		
		t0=time()
		total_loss = torch.Tensor([0])
		for root in train_set:
			#t00=time()
			model.zero_grad()
			model(root,embedding_dim)
		
			loss=model.TreeRNN_MSE(root,embedding_dim)
			total_loss+=loss.data
		
			loss.backward()
			optimizer.step()
			#print(time()-t00)
			#k=input()	
		print("Epoch number {0}: \nTime : {1:0.4f}\nError: {2:0.4f}\nAverage Error per sentence: {3:0.4f}".format(epoch,(time()-t0),total_loss[0],total_loss[0]/len(train_set)),end="\n\n")	
		losses.append(total_loss[0])
		epochs.append(epoch)
		
		
	torch.save(model,"Saved Models/Saved_model_"+str(lr)+"_"+str(len(epochs)+1)+".pt")
	torch.save(model,"Saved Models/W2V_"+str(lr)+"_"+str(len(epochs)+1)+".pt")	
	plt.title("Mean Squared Error loss over training dataset")
	plt.xlabel("Epochs")
	plt.ylabel("MSE Loss")
	plt.plot(epochs,losses,label = "Mean Squared Error Loss")
	plt.legend(loc='upper right')
	plt.show()
	print("Time taken to train: {0:0.4f}".format(time()-total_time))
	#print(epochs)
	#print(losses)
	#print(embed(Variable(torch.LongTensor([words2idx['the']]))))
	
word_vec_file="GloVe/glove.6B/glove.6B.100d.txt"
embedding_dim=100
embed,words2idx=getEmbeddings(word_vec_file,embedding_dim)
#print(embed(Variable(torch.LongTensor([words2idx['the']]))))
labels=['compound', 'advmod', 'cc_preconj', '\n', 'mark', 'acl', 'auxpass', 'ccomp', 'conj', 'cc', 'parataxis', 'punct', 'nmod_poss', 'det_predet', 'nsubjpass', 'case', 'nsubj', 'expl', 'iobj', 'det', 'csubjpass', 'neg', 'discourse', 'amod', 'mwe', 'xcomp', 'nummod', 'compound_prt', 'cop', 'dep', 'nmod_tmod', 'root', 'nmod_npmod', 'dobj', 'appos', 'nmod', 'aux', 'csubj', 'acl_relcl', 'advcl']

dataset=generate_dataset()
train_set=dataset[0:100]
missing_words=[]
lr=0.001
epochs=100000
train_model(epochs,train_set,word_vec_file,embedding_dim,lr)
model=torch.load("Saved Models/Saved_model_"+str(lr)+"_"+str(epochs+1)+".pt")
#train_set[0].print_tree()
model(train_set[0],embedding_dim)
train_set[0].print_tree()
print(model.TreeRNN_MSE(train_set[0],embedding_dim))
print(len(set(missing_words)))
print(set(missing_words))
