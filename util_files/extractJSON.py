import json
f=open("Input Files/questions_and_answers_ntu.json","r")
f2=open("Input Files/NTU_ques_dependencies_parsed.txt","w")
db=json.load(f)
keys=db.keys()
for k in keys:
	f2.write(db[k]['question'])
	f2.write("\n")
f.close()
f2.close()
