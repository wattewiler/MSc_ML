import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Vorbereitung der Daten
df_cleaned = pd.read_csv('D:/Data_kDrive/_Studium/MSc Applied Information and Data Science/MyStuff/_Masterthesis/_MeineDaten/01_student_data_cleaned.csv')
X = df_cleaned.drop(['grade'], axis=1)  # Entferne die Spalten G1, G2 und benutze G3 als Zielvariable
y = df_cleaned['grade']

# Identifiziere kategoriale und numerische Spalten
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

# Preprocessing für numerische und kategoriale Daten
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Aufteilen in Trainings- und Testdatensätze
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Erstellen und Konfigurieren des XGBoost-Modells
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42))])

# Trainiere das Modell
model.fit(X_train, y_train)

# Vorhersagen treffen
y_pred = model.predict(X_test)

# Leistungsmetriken berechnen und ausgeben
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
