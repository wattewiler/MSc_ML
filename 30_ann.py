import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Daten laden
df_cleaned = pd.read_csv('D:/Data_kDrive/_Studium/MSc Applied Information and Data Science/MyStuff/_Masterthesis/_MeineDaten/01_student_data_cleaned.csv')

# Vorbereitung der Daten
X = df_cleaned.drop('grade', axis=1)
y = df_cleaned['grade']

# Identifizieren kategorialer und numerischer Spalten
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Modell definieren
model = Sequential([
    # Erste Schicht: Anzahl der Neuronen und Aktivierungsfunktion anpassen
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    # Dropout-Schicht zur Reduzierung von Overfitting
    Dropout(0.1),
    # Weitere Schichten hinzufügen/entfernen oder die Anzahl der Neuronen/Aktivierungsfunktionen ändern
    Dense(64, activation='relu'),
    # Ausgabeschicht: Für Regression normalerweise eine Neuron mit linearer Aktivierung
    Dense(1, activation='linear')
])

# Optimierer: Lernrate und Optimierer-Typ anpassen
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Modell trainieren: Epochen und Batch-Größe anpassen
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Modell bewerten
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")