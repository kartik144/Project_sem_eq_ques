import os
import nltk
import networkx as nx
from subprocess import check_call

class TreeNode:
	def __init__(self,name):
		self.children=[]
		self.data=name
		self.X=None
		self.H=None
		self.U=None#############CHANGE
		self.R=None
		self.type=None
		self.parent=None
		return
		
	def addNode(self, node, ty):
		assert isinstance(node,TreeNode)
		node.type=ty
		node.parent=self
		(self.children).append(node)
		return 
		
	def generate_parse_Tree(self,nodes):
		for child,ty in nodes[self.data]:
			ch=TreeNode(child)
			self.addNode(ch,ty)
			if child in nodes.keys():
				ch.generate_parse_Tree(nodes)	
		return
	
	def print_tree(self):
		print(self.data,end=" :: ")
		for child in self.children:
			print("("+child.data+","+child.type+")",end="  ")
		print()
		print(self.X)
		print(self.R)
		for child in self.children:
			child.print_tree()
		return
		
def get_input_file_data(input_file):
	f=open(input_file,"r")
	sen=f.readline()
	tree_data=[]
	nodes={}
	while sen!="":
		tree_data.append(sen)
		sen=f.readline()
	
	for s in tree_data:
		ty=(s.split('(')[0]).replace(":","_")
		from_node=(s.split('(')[1]).split(',')[0]
		to_node=((s.split('(')[1]).split(',')[1])[1:-2]
		
		if from_node not in nodes.keys():
			nodes[from_node]=[]
		nodes[from_node].append((to_node,ty))
		
	return nodes

def print_parse_tree(nodes):
	edge_labels={}
	G=nx.DiGraph()
		
	for k in nodes.keys():
		for j,t in nodes[k]:
			G.add_edge(k,j,label=t)
		
	nx.drawing.nx_pydot.write_dot(G,os.getcwd()+"/Graphs/"+"parse_tree.dot")	
	check_call(['dot','-Tpng',os.getcwd()+"/Graphs/parse_tree.dot",'-o',os.getcwd()+"/Graphs/"+"parse_tree.png"])
	
	return nodes
	
def generate_Tree(input_file="Input Files/input_file_dependency.txt"):
	nodes=get_input_file_data(input_file)
	print_parse_tree(nodes)
	root=TreeNode('ROOT-0')
	root.generate_parse_Tree(nodes)
	return root
	
#root=generate_Tree()
#root.print_tree()

#root=TreeNode('A')
#root.X=5
#root.R=2

