# Modellevaluation zur Vorhersage von Schülerleistungen

Dieses Repository enthält eine Sammlung von Python-Skripten, die verschiedene Machine Learning-Modelle zur Vorhersage der Leistung von Schülern anhand eines Datensatzes implementieren. Jedes Skript ist auf eine spezifische Aufgabe innerhalb des Data Science Workflows ausgerichtet, von der Datenvorbereitung bis zum Training und Tuning von Modellen.

Dieser Code wurde während des MSc in Applied Information and Data Science an der HSLU geschrieben.

## Skriptübersicht

- **00_load_prep_and_data_exploration.py**: Lädt den Datensatz, bereitet die Daten vor (inklusive Bereinigung und Vorverarbeitung) und führt eine explorative Datenanalyse durch.
- **10_linear_regression.py**: Implementiert ein lineares Regressionsmodell zur Vorhersage der Schülerleistungen.
- **20_xgboost.py**: Nutzt das XGBoost-Modell für die Vorhersage, ohne Hyperparameter-Tuning.
- **25_xgboost_with_hp.py**: Erweitert das XGBoost-Modell durch Hyperparameter-Tuning mit `hyperopt`.
- **30_ann.py**: Erstellt und trainiert ein einfaches künstliches neuronales Netzwerk (ANN) zur Vorhersage.
- **35_ann_with_hp.py**: Implementiert ein ANN mit Hyperparameter-Tuning unter Verwendung von Keras Tuner.

## Voraussetzungen

Um die Skripte auszuführen, sind folgende Libraries erforderlich:

- pandas
- scikit-learn
- xgboost
- tensorflow / keras
- keras-tuner (für `35_ann_with_hp.py`)
- hyperopt (für `25_xgboost_with_hp.py`)

Diese Libraries können über pip installiert werden:

```bash
pip install pandas scikit-learn xgboost tensorflow keras-tuner hyperopt
