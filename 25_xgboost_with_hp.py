import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# Vorbereitung der Daten
df_cleaned = pd.read_csv('D:/Data_kDrive/_Studium/MSc Applied Information and Data Science/MyStuff/_Masterthesis/_MeineDaten/01_student_data_cleaned.csv')
X = df_cleaned.drop(['grade'], axis=1)
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

# Definiere den Suchraum
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'max_depth': hp.choice('max_depth', range(3, 11)),
    'subsample': hp.uniform('subsample', 0.7, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1),
}

# Ziel-Funktion für Hyperopt
def objective(space):
    model = xgb.XGBRegressor(
        n_estimators=int(space['n_estimators']),
        learning_rate=space['learning_rate'],
        max_depth=int(space['max_depth']),
        subsample=space['subsample'],
        colsample_bytree=space['colsample_bytree'],
        objective='reg:squarederror',
        eval_metric="rmse"
    )
    
    # Pipeline mit dem XGBoost-Modell
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    pipeline.fit(X_train, y_train)
    
    pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return {'loss': mse, 'status': STATUS_OK}

# Starte das Tuning
trials = Trials()
best_hyperparams = fmin(fn=objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=100,  # Anzahl der Evaluierungen
                        trials=trials)

print("Die besten Hyperparameter sind: ", best_hyperparams)