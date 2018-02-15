f=open("Input Files/NTU_ques_dependencies_parsed.txt","r")
labels=[]
s=f.readline()
while (s != ""):
	ty=(s.split('(')[0]).replace(":","_")
	labels.append(ty)
	s=f.readline()

print(list(set(labels)))
