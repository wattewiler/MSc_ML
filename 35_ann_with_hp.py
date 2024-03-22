import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner import RandomSearch

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

# Funktion zur Definition des Modells mit Hyperparameter-Suchraum
def build_model(hp):
    model = Sequential([
        Dense(units=hp.Int('units_input', min_value=32, max_value=512, step=32),
              activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(rate=hp.Float('dropout_input', min_value=0.0, max_value=0.5, step=0.1)),
        Dense(units=hp.Int('units_hidden', min_value=32, max_value=512, step=32),
              activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mse',
                  metrics=['mae'])
    return model

# Konfigurieren des Tuners
tuner = RandomSearch(
    build_model,
    objective='val_mae',
    max_trials=10,
    executions_per_trial=1,
    directory='keras_tuner',
    project_name='student_grade_prediction'
)

# Starten des Tuning-Prozesses
tuner.search(X_train, y_train, epochs=20, validation_split=0.2)

# Ausgabe der besten Hyperparameter
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Beste Hyperparameter: {best_hps.values}")

# Abrufen und Evaluieren des besten Modells
best_model = tuner.get_best_models(num_models=1)[0]
loss, mae = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")
