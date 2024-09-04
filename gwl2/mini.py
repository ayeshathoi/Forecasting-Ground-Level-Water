import pandas as pd
import seaborn as sb

data=pd.read_csv("data.csv")
data.head()
sb.countplot(x='Situation',data=data,palette='bright')
X = data.iloc[:, [4,5,6,9,10,11]]
# X --> 
# Total_Rainfall,Natural discharge during non-monsoon season,
# Net annual groundwater availability,
# Groundwater availability for future irrigation use


Y = data.iloc[:,12]
# Y --> Situation


from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder() # Label Encoder to encode the target variable 
# It will convert the string values of the target variable to integers : Excess = 1, Moderate = 3
Y = labelencoder_Y.fit_transform(Y) 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # Standard Scaler to scale the features to give each feature equal importance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
from sklearn.metrics import accuracy_score
score =accuracy_score(Y_test,y_pred)

Availabilty = pd.get_dummies(data['Situation'],drop_first=True) # One hot encoding for the target variable
data.drop(['Situation'],axis=1,inplace=True) # Drop the target variable from the data
data1 = pd.concat([data,Availabilty],axis=1) # Concatenate the one hot encoded target variable to the data
X = data1.iloc[:, [4,5,6,9,10,11]]
Y = data1.iloc[:,12]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
new_score=classifier.score(X_test, Y_test)


# Level encoding vs One hot encoding
print("Accuracy Score in test size 1/3: ",score)
print("New Score in test size .55",new_score)
print("Difference: ",new_score-score)
