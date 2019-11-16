
import numpy as np
import pandas as pd

input_file = "TRAINING.csv"
test_file="TEST.csv"

df = pd.read_csv(input_file, header = 0)
dft = pd.read_csv(test_file, header = 0)

#cleaning training
df['Grade'] = df['Grade'].map({'A': 0, 'B': 1,'C': 2, 'D': 3, 'E':4})
df['roof']=df['roof'].str.upper()
df['roof'] = df['roof'].map({'NO': 0, 'YES': 1})
df['Price'] = df['Price'].map(lambda x:int(x.rstrip('$')))

#cleaning testing
dft['roof']=dft['roof'].str.upper()
dft['roof'] = dft['roof'].map({'NO': 0, 'YES': 1})
dft['Price'] = dft['Price'].map(lambda x:int(x.rstrip('$')))


#replacing null values
df.fillna(df.mean(), inplace=True)
dft.fillna(dft.mean(), inplace=True)

# X -> features, y -> label 
X = df._drop_axis(['Grade','id'],axis=1)
y = df['Grade']

#test data
y_train=dft._drop_axis(['id'],axis=1)


X_train, X_test= X,y


from sklearn.svm import SVC
svm = SVC(kernel="linear",C=0.005,random_state=101)
svm.fit(X,y)
v=svm.predict(y_train)
print(v)
import csv
dic={0:'A',1:'B',2:'C',3:'D',4:'E'}
csvfile="E:\\hackathon\\geeks\\testingp.csv"
row1=[["id","Grade"]]
with open(csvfile,"w",newline='') as f:
    writer=csv.writer(f)
    writer.writerows(row1)
    i=0
    for val in v:
        i+=1
        line=[[i,dic[val]]]
        writer.writerows(line)
    
