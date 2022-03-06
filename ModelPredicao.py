#Cria modelos de predição

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle

dataframe = pd.read_csv('/Dataset/train_final.csv')
y = dataframe['level']
x = dataframe.iloc['virgula', 'quantPalavras', 'canonicas', 'quantStopWords']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

SVC_model = svm.SVC()
KNN_model = KNeighborsClassifier(n_neighbors=5)
SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)
SVC_prediction = SVC_model.predict(X_test)
KNN_prediction = KNN_model.predict(X_test)

print(accuracy_score(SVC_prediction, y_test))
print(accuracy_score(KNN_prediction, y_test))

print(confusion_matrix(SVC_prediction, y_test))
print(classification_report(KNN_prediction, y_test))

saved_model_knn = pickle.dumps(KNN_model)
saved_model_svc = pickle.dumps(SVC_model)

file_KNN = open('Models/KNN_model.txt', 'w')
file_SVC = open('Models/KNN_model.txt', 'w')

file_KNN.write(save_model_knn)
file_SVC.write(save_model_svc)

file_KNN.close()
file_SVC.close()