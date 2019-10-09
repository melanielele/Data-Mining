'''this code is written by Melanie Zhao and Cynthia without consulting others except TAs and instructor
'''
import sys

def loadTrainingSet():#load dataset based on the first input command line
	with open(sys.argv[1], 'r') as file:
		 dataset=[[str(x) for x in line.split()]for line in file]
		 print('successful load training data')
		 #print(range(len(dataset)))
	return dataset


def computeProbs(dataset):   # Return two lists that contains every possibility that may be used
	column_value = {}#use this dictionary to store unique elements in each column(attribute values)
	for i in range(1,len(dataset[0])):#dataset[0] are the attributes
		column_value[i] = list(set(map(lambda tuple: tuple[i], dataset)))
		#print(len(column_value[i]))
	classes = list(set(map(lambda tuple: tuple[0], dataset)))  #class is the 
	#print(classes)
	#print(len(dataset))
	class_subsets = []#store the tuples that associated with certian class(p or e)
	P_Ci= dict()
	for i in range(len(classes)):
		subset = filter(lambda x: x[0] == classes[i], dataset)
		#print(subset)
		class_subsets.append(subset)
		P_Ci[classes[i]] = len(subset) / float(len(dataset)) #count how many times each class occur
		#this is the P(Ci)
	#print(P_Ci) {'p': 0.5454545454545454, 'e': 0.45454545454545453}
	#print(P_Ci[list(P_Ci.keys())[0]])
	dicts = []   #list of dictionaries
	#Designed structure is like[classdescrip:{attributeLIST}]
	#dict[0] = p,dict[1] = e ---classdictionary
	#which store all the posteriori probability of each class
	for i in range(len(class_subsets)):
		subset = class_subsets[i] #each class contains different tuples 
		#use a dict to store different(P.xk|Ci) so that we can compute the D P.x1jCi/P.x2jCi/  P.xnjCi/.
		dicts.append(dict())
		ClassDescrip = dicts[i]#each class has a classdescrip
		for column in range(1,len(dataset[0])):
			AttributeList = ClassDescrip[column] = {}  #in each column that represent an Attibute, it contains different value, we need to count the occurance of different values 
			for tuple in subset:
				val= tuple[column]
				if val not in AttributeList.keys():
					AttributeList[val] = 1
				else:
					AttributeList[val] +=1#store the value occur in certain attribute 
			#print(AttributeList)
			#print(AttributeList.keys())
			for value in AttributeList.keys():
				AttributeList[value] += 1
				#print(AttributeList[value])			
				AttributeList[value]/= float(len(subset)+len(column_value[column]))#Laplacian correction
				#print(AttributeList[value])	
			for value in column_value[column]:
				if value not in AttributeList.keys():
					AttributeList[value] = 1/float(len(subset)+len(column_value[column])) #Laplacian correction

			 
	#print(dicts)
	return P_Ci, dicts
	
def Classify(Testdataset,dicts,P_Ci):
	with open(Testdataset, 'r') as file:
		Testdata=[[str(x) for x in line.split()]for line in file]
		print('successful load test data')
	#print(Testdata)
	Classlabel=[]
	for tuple in Testdata:#each tuple we need to classify it
		Px=[] #list to store p(X|Ci)
		for i in range(len(P_Ci)):#2 classes here 
			Px.append(float(1))
			for j in range(1,len(tuple)):#everyattribute,column 
				attribute=tuple[j]
				Px[i] *= dicts[i][j][attribute]
			#after multiply every attribute possibility,multiply the pci
			Px[i]*=P_Ci[list(P_Ci.keys())[i]]
		maxPx = max(Px)
		print(Px)
		classnum = Px.index(maxPx)
		#add classlabel to the classlabel list
		tuple.append(list(P_Ci.keys())[classnum])
		Classlabel.append(list(P_Ci.keys())[classnum])
	#print(Testdata)
	return Testdata, Classlabel



def Acutaldata(Testdataset,P_Ci):
	actualTrue = 0
	actualFalse = 0
	#assume P_Ci[0] = true, P_Ci[1] = false;
	Truecase = list(P_Ci.keys())[0]
	Flasecase = list(P_Ci.keys())[1]
	for tuple in Testdataset:
		if tuple[0] == Truecase:
			actualTrue+=1
		else:
			actualFalse+=1
	print('actualTrue',actualTrue)
	print('actualFalse',actualFalse)
	return actualTrue, actualFalse

def Predicteddata(classlabel,P_Ci):
	predictTrue = 0
	predictFalse = 0
	#assume P_Ci[0] = true, P_Ci[1] = false;
	Truecase = list(P_Ci.keys())[0]
	Flasecase = list(P_Ci.keys())[1]

	for item in classlabel:
		if item == Truecase:
			predictTrue+=1
		else:
			predictFalse+=1
	print('predictTrue',predictTrue)
	print('predictFalse',predictFalse)
	return predictTrue,predictFalse

def MatrixValue(Testdata,P_Ci):
	TruePos = 0
	TrueNeg = 0
	Truecase = list(P_Ci.keys())[0]
	Flasecase = list(P_Ci.keys())[1]

	for tuple in Testdata:
		if tuple[0] == Truecase:
			if tuple[0]==tuple[-1]:
				TruePos +=1
		else:
			if tuple[0]==tuple[-1]:
				TrueNeg +=1
	print('TruePos',TruePos)
	print('TrueNeg',TrueNeg)
	return TruePos, TrueNeg

def main():
	L=loadTrainingSet()
	#Classify(L)
	PCi, dicts = computeProbs(L)
	Testdata,Label = Classify(sys.argv[2],dicts,PCi)
	file = open(sys.argv[3],"w+")
	for classlabel in Label:
		file.write(classlabel)
		file.write('\n')
	file.close()
	actualTrue, actualFalse = Acutaldata(Testdata,PCi)
	predictTrue,predictFalse = Predicteddata(Label,PCi)
	TruePos, TrueNeg = MatrixValue(Testdata,PCi)
	FalsePos = actualFalse - TrueNeg
	FalseNeg = actualTrue - TruePos
	print('FalsePos',FalsePos)
	print('FalseNeg',FalseNeg)
	total = actualTrue+actualFalse
	Accuracy = (TruePos+TrueNeg)/float(total)
	print('Accuracy',Accuracy)
	Recall = TruePos/float(actualTrue)
	print('Recall',Recall)
	Precision = TruePos/float(predictTrue)
	print('Precision',Precision)

	file = open(sys.argv[4],"w+")
	file.write('Accuracy: ')
	file.write(str(Accuracy))
	file.write('\n')
	file.write('Recall: ')
	file.write(str(Recall))
	file.write('\n')
	file.write('Precision: ')
	file.write(str(Precision))
	file.write('\n')
	file.write('Confusion Matrix:')
	file.write('\n')
	file.write('TrueNeg: ')
	file.write(str(TrueNeg))
	file.write('  ')
	file.write('FalsePos: ')
	file.write(str(FalsePos))
	file.write('  ')
	file.write('\n')
	file.write('FalseNeg: ')
	file.write(str(FalseNeg))
	file.write('  ')
	file.write('TruePos: ')
	file.write(str(TruePos))
	file.close()




if __name__ == '__main__':
	main()
