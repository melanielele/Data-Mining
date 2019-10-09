'''this code is written by Melanie Zhao and Qihang Zhang without consulting others except TAs and instructor

https://scikit-learn.org/stable/modules/ensemble.html#majority-class-labels-majority-hard-voting

    '''

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn import svm #Algorithm C
from sklearn import tree #Algorithm A
from sklearn.naive_bayes import GaussianNB #Algorithm B
from sklearn.ensemble import VotingClassifier #ensemble algorithm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score, accuracy_score, f1_score,confusion_matrix

data = pd.read_csv(sys.argv[1])
#print(data.head())
#print(data.shape)#give us information about how many students and attributes 




def define_Letter(df):
	letters = [] #list of letter based on the letter grade
	for row in df['Total Score']:
		if row >=900.0 :
			letters.append('A')
		elif (row <900.0 and row >=800.0):
			letters.append('B')
		elif (row <800.0 and row >=700.0):
			letters.append('C')
		elif (row <700.0 and row >=600.0):
			letters.append('D')
		elif (row <600.0):
			letters.append('F')
	df['Letter']= letters
	return df

#create new attribute
data = define_Letter(data)

 #fill all blank with 0s
data.replace(r'\s+',np.nan,regex=True).replace('',np.nan)
data = data.fillna(0)



def classification(features):
	X = features
	y = np.array(data['Letter'])

	#LOO cross validatio
	kf = KFold(n_splits=2)
	for train, test in kf.split(X):
		#print("%s %s" % (train, test))
		X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
		#print(X_train, X_test, y_train, y_test)
	
	model1 = tree.DecisionTreeClassifier()
	model2 = GaussianNB()
	model3 = svm.SVC(gamma='scale',decision_function_shape='ovo')
	emodel = VotingClassifier(estimators=[('decisionTree',model1),('NB',model2),('svm',model3)],voting='hard')
	emodel.fit(X_train, y_train)
	y_predict = emodel.predict(X_test)
	'''
	for model, label in zip([model1, model2, model3, emodel], ['Decision Tree', 'Naive Bayes', 'SVM', 'Ensemble']):
		scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
		print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
	'''
	print('accuracy')
	print(accuracy_score(y_test, y_predict))
	print('f1')
	print(f1_score(y_test, y_predict,average='weighted'))
	print('recall')
	print(recall_score(y_test, y_predict,average='weighted'))
	print('precision')
	print(precision_score(y_test, y_predict,average='weighted'))
	print(confusion_matrix(y_test, y_predict))
print("Ensemble Method-voting")
print('1')
X1 =np.array(data[['Quiz 01']])#2D array
classification(X1)
print('2')
X2 =np.array(data[['Quiz 01','Quiz 02']])
classification(X2)
print('3')
X3 =np.array(data[['Quiz 01','Quiz 02','Quiz 03']])
classification(X3)
print('4')
X4 =np.array(data[['Quiz 01','Quiz 02','Quiz 03','Quiz 04']])
classification(X4)
print('5')
X5 =np.array(data[['Quiz 01','Quiz 02','Quiz 03','Quiz 04','Quiz 05']])
classification(X5)
print('6')
X6 =np.array(data[['Quiz 01','Quiz 02','Quiz 03','Quiz 04','Quiz 05','Quiz 06']])
classification(X6)
print('7')
X7 =np.array(data[['Quiz 01','Quiz 02','Quiz 03','Quiz 04','Quiz 05','Quiz 06','Quiz 07']])
classification(X7)
print('8')
X8 =np.array(data[['Quiz 01','Quiz 02','Quiz 03','Quiz 04','Quiz 05','Quiz 06','Quiz 07','Quiz 08']])
classification(X8)
print('9')
X9 =np.array(data[['Quiz 01','Quiz 02','Quiz 03','Quiz 04','Quiz 05','Quiz 06','Quiz 07','Quiz 08','Quiz 09']])
classification(X9)
print('10')
X10 =np.array(data[['Quiz 01','Quiz 02','Quiz 03','Quiz 04','Quiz 05','Quiz 06','Quiz 07','Quiz 08','Quiz 09','Quiz 10']])
classification(X10)
print('11')
X11 =np.array(data[['Quiz 01','Quiz 02','Quiz 03','Quiz 04','Quiz 05','Quiz 06','Quiz 07','Quiz 08','Quiz 09','Quiz 10','Quiz 11']])
classification(X11)
print('12')
X12 =np.array(data[['Quiz 01','Quiz 02','Quiz 03','Quiz 04','Quiz 05','Quiz 06','Quiz 07','Quiz 08','Quiz 09','Quiz 10','Quiz 11','Quiz 12']])
classification(X12)



