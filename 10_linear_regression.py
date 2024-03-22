import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df_cleaned = pd.read_csv('D:/Data_kDrive/_Studium/MSc Applied Information and Data Science/MyStuff/_Masterthesis/_MeineDaten/01_student_data_cleaned.csv')


# Identifizieren kategorialer und numerischer Spalten
categorical_features = df_cleaned.select_dtypes(include=['object']).columns
numerical_features = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.drop('grade')

# Erstellen des Preprocessing-Pipelines für numerische und kategoriale Daten
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Definieren des Modells
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Aufteilung in Trainings- und Testdatensätze
X = df_cleaned.drop('grade', axis=1)
y = df_cleaned['grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell trainieren
model.fit(X_train, y_train)

# Vorhersagen treffen
y_pred = model.predict(X_test)

# Leistungsmetriken berechnen
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")