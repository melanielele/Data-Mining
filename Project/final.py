
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score, accuracy_score, f1_score,confusion_matrix
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
#Data part
def cap(x, quantile=[0.01, 0.99]): 
	Q01, Q99 = x.quantile(quantile).values.tolist()

	#we replace the outliers out of 0.01 and 0.99 using the quantile values
	if Q01 > x.min():
		x = x.copy()
		x.loc[x < Q01] = Q01

	if Q99 < x.max():
		x = x.copy()
		x.loc[x > Q99] = Q99

	return (x)


dt = pd.read_csv("heart.csv") #load the csv file
dt.drop_duplicates()# here we want to drop all duplicate data
dt.isnull().values.any()#all rows control for null values, we don't have any null value for this data
dt = dt.apply(cap,quantile=[0.01,0.99]) # clear outliers using capping method

#then we do the grouping of ages
young_ages=dt[(dt.age>=29)&(dt.age<40)]
middle_ages=dt[(dt.age>=40)&(dt.age<55)]
elderly_ages=dt[(dt.age>55)]
sns.barplot(x=['young ages','middle ages','elderly ages'],y=[len(young_ages),len(middle_ages),len(elderly_ages)])
plt.xlabel('Age Range')
plt.ylabel('Age Counts')
plt.title('Ages State in Dataset')
plt.show()

young_andtarget_on=len(dt[(dt.age>=29)&(dt.age<40)& (dt['target'] == 1)])#means yes
young_andtarget_off=len(dt[(dt.age>=29)&(dt.age<40)& (dt['target'] == 0)])
middle_andtarget_on=len(dt[(dt.age>=40)&(dt.age<55)& (dt['target'] == 1)])
middle_andtarget_off=len(dt[(dt.age>=40)&(dt.age<55)& (dt['target'] == 0)])
elderly_andtarget_on=len(dt[(dt.age> 55) & (dt['target'] == 1)])
elderly_andtarget_off=len(dt[(dt.age> 55) & (dt['target'] == 0)])
####
sns.barplot(x=['Young Target On','Young Target Off','Middle Target On','Middle Target Off','Elderly Target On','Elderly Target Off'],
			y=[young_andtarget_on,young_andtarget_off,middle_andtarget_on,middle_andtarget_off,elderly_andtarget_on,elderly_andtarget_off])
plt.xlabel('Age group and Target State')
plt.ylabel('Count')
plt.title('State of the Age')
plt.show()

#then we are going to do the stats based on gender
sns.barplot(x=['Male','Female'],y=[len(dt[dt['sex'] == 1]),len(dt[dt['sex'] == 0])])
plt.show()

# we calculate how many male and female are on state
male_andtarget_on=len(dt[(dt.sex==1)&(dt['target']==1)])
male_andtarget_off=len(dt[(dt.sex==1)&(dt['target']==0)])
female_andtarget_on=len(dt[(dt.sex==0)&(dt['target']==1)])
female_andtarget_off=len(dt[(dt.sex==0)&(dt['target']==0)])
####
sns.barplot(x=['Male Target On','Male Target Off','Female Target On','Female Target Off'],
			y=[male_andtarget_on,male_andtarget_off, female_andtarget_on,female_andtarget_off])
plt.xlabel('Male and Female target State')
plt.ylabel('Count')
plt.title('State of the Gender')
plt.show()

#Age vs Thalch
age_unique=sorted(dt.age.unique())
age_thalach_values=dt.groupby('age')['thalach'].count().values
mean_thalach=[]
for i,age in enumerate(age_unique):
	mean_thalach.append(sum(dt[dt['age']==age].thalach)/age_thalach_values[i])
plt.figure(figsize=(10,5))
sns.pointplot(x=age_unique,y=mean_thalach,color='red',alpha=0.8)
plt.xlabel('Age',fontsize = 15,color='blue')
plt.xticks(rotation=45)
plt.ylabel('Thalach',fontsize = 15,color='blue')
plt.title('Age vs Thalach',fontsize = 15,color='blue')
plt.grid()
plt.show()


dataX=dt.drop('target',axis=1)
dataY=dt['target']

#model part
x=dt.drop('target', 1)
y=dt['target']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = .2, random_state=0) 
#split the data to train and test
#80%----  train data and 20% ---test data.

#we choose naive bayes algorithm to fit the model
model = GaussianNB()
y_predict = model.fit(X_train, y_train).predict(X_test)
print("Naive Bayes")
print('accuracy')
print(accuracy_score(y_test, y_predict))

print('f1')
print(f1_score(y_test, y_predict,average='weighted'))

print('recall')
print(recall_score(y_test, y_predict,average='weighted'))

print('precision')
print(precision_score(y_test, y_predict,average='weighted'))

print(confusion_matrix(y_test, y_predict))

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("decision tree")
print('accuracy')
print(accuracy_score(y_test, y_predict))

print('f1')
print(f1_score(y_test, y_predict,average='weighted'))

print('recall')
print(recall_score(y_test, y_predict,average='weighted'))

print('precision')
print(precision_score(y_test, y_predict,average='weighted'))

print(confusion_matrix(y_test, y_predict))


#visulize decision tree
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png("tree.png")
#correlation part
corr = x.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,mask=mask,cmap='summer_r',vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
