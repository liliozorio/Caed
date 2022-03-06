#Cria modelos de predição

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
import pickle

dataframe = pd.read_csv('/content/drive/MyDrive/Prova Caed/train_final.csv')
y = dataframe['level']
x = dataframe[['virgula', 'quantPalavras', 'canonicas', 'quantStopWords']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=20)

SVC_model = svm.SVC()
KNN_model = KNeighborsClassifier(n_neighbors=5)
SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)
SVC_prediction = SVC_model.predict(X_test)
KNN_prediction = KNN_model.predict(X_test)

print(accuracy_score(SVC_prediction, y_test))
print(accuracy_score(KNN_prediction, y_test))

print(confusion_matrix(SVC_prediction, y_test))
print(confusion_matrix(KNN_prediction, y_test))
print(classification_report(SVC_prediction, y_test))
print(classification_report(KNN_prediction, y_test))

saved_model_knn = pickle.dumps(KNN_model)
saved_model_svc = pickle.dumps(SVC_model)

file_SVC = open('/content/drive/MyDrive/Prova Caed/SVC_model.sav', 'wb')
file_KNN = open('/content/drive/MyDrive/Prova Caed/KNN_model.sav', 'wb')
file_SVC.write(saved_model_svc)
file_KNN.write(saved_model_knn)
file_SVC.close()
file_KNN.close()
