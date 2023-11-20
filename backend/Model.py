import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Charger les données depuis le fichier CSV
data = pd.read_csv('customer_churn.csv')

# Supprimer les colonnes non pertinentes ou uniques pour chaque entrée
columns_to_drop = ['Names', 'Onboard_date', 'Location', 'Company']
data = data.drop(columns=columns_to_drop)

# Séparer les caractéristiques (features) et la cible (target)
X = data.drop('Churn', axis=1)
y = data['Churn']

model = ExtraTreesClassifier()
model.fit(X, y)
print(data.head())

# Obtenir des informations sur les types de données et les valeurs manquantes
print(data.info())

# Statistiques descriptives pour les colonnes numériques
print(data.describe())

print(model.feature_importances_)

# data = data.drop(columns='Account_Manager')
# print(data.head())

# X = data.drop('Churn', axis=1)
# y = data['Churn']
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les caractéristiques numériques (MinMaxScaler pour mettre à l'échelle entre 0 et 1)
scaler = MinMaxScaler()
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(X_train)
xtest = sc_x.transform(X_test)


print(  xtrain)
print(  "xxxxxxxxxxxxx test ", xtest)

classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, y_train)

y_pred = classifier.predict(xtest)
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix  LogisticRegression : \n", cm)
print ("Accuracy LogisticRegression : ", accuracy_score(y_test, y_pred))


#/////////////////////////////////////////////////////////////////////////////////
# Create KNN classifier
from sklearn.neighbors import KNeighborsClassifier

acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
   neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
   yhat = neigh.predict(X_test)
   acc.append(metrics.accuracy_score(y_test, yhat))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed',
    marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))


knn = KNeighborsClassifier(n_neighbors=5)
# Train the model using the training sets
knn.fit(xtrain, y_train)

y_pred = knn.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix  KNeighborsClassifier: \n", cm)
# Model Accuracy, how often is the classifier correct?
print("Accuracy KNeighborsClassifier:",metrics.accuracy_score(y_test, y_pred))

#/////////////////////////////////////////////////////////////////////////////////
# Créer et entraîner le modèle SVM non linéaire avec noyau RBF
svm_non_linear = SVC(kernel='rbf')
svm_non_linear.fit(xtrain, y_train)

# Faire des prédictions
predictions_non_linear = svm_non_linear.predict(X_test)

# Évaluer la précision du modèle
accuracy_non_linear = accuracy_score(y_test, predictions_non_linear)
print(f"Précision du SVM non linéaire (RBF) : {accuracy_non_linear}")

from joblib import dump

# Enregistrer le meilleur modèle SVM sous format .pkl
dump(classifier, 'meilleur_modele_svm.pkl')
