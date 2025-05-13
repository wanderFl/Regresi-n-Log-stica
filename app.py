# Importar librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Cargar el dataset
df = pd.read_csv('AirPassengers.csv')

# Convertir la columna 'Month' a tipo datetime
df['Month'] = pd.to_datetime(df['Month'])

# Crear la variable binaria 'Aumento'
df['Aumento'] = df['#Passengers'].diff().apply(lambda x: 1 if x > 0 else 0)

# Eliminar la primera fila (que tiene NaN en 'Aumento')
df = df.dropna().reset_index(drop=True)

# Agregar características adicionales
df['MesIndice'] = range(len(df))
df['MesDelAño'] = df['Month'].dt.month
df['Año'] = df['Month'].dt.year

# Separar variables predictoras y objetivo
X = df[['MesIndice', 'MesDelAño', 'Año']].values
y = df['Aumento'].values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Escalar características
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrenar el modelo de regresión logística
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predecir sobre el conjunto de prueba
y_pred = classifier.predict(X_test)

# Evaluar el modelo
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Visualización de la matriz de confusión
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Aumenta', 'Aumenta'], yticklabels=['No Aumenta', 'Aumenta'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title(f'Matriz de Confusión\nPrecisión del Modelo: {accuracy:.2%}')
plt.tight_layout()
plt.show()
