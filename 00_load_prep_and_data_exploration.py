##### Setup
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

sns.set(style="whitegrid")

#### Import der Daten
base_path = 'D:/Data_kDrive/_Studium/MSc Applied Information and Data Science/MyStuff/_Masterthesis/_MeineDaten/'

original_data_path = base_path + '01_student_data.csv'
cleaned_data_path = base_path + '01_student_data_cleaned.csv'

df = pd.read_csv(original_data_path)

#### Data Preperation
# Droppen der Kolonnen G1 und G2, da der Fokus auf G3 (Jahresnoten) liegt
df.drop(['G1', 'G2'], axis=1, inplace=True)                             

# Umbenennen der Spalte G3 in "grade"
df.rename(columns={'G3': 'grade'}, inplace=True)

#### Data Exploration
# Berechnung der fehlenden Daten
missing_data_counts = df.isnull().sum()
missing_data_counts = missing_data_counts[missing_data_counts > 0]

# Visualisierung der fehlenden Daten anhand missingno libary
msno.bar(df)
msno.matrix(df)

# Statistische Zusammenfassung des Datasets
df_info = df.describe()

# Weitere Visualisierung der fehlenden Daten
plt.figure(figsize=(10, 6), dpi=300)
missing_data_counts.plot(kind='bar')
plt.title('Anzahl fehlender Werte pro Spalte')
plt.ylabel('Anzahl fehlender Werte')
plt.xlabel('Spaltennamen')
missing_values_fig = plt.gcf()

# Droppen aller Zeilen mit fehlenden Werten
df_cleaned = df.dropna()

##### Visulisierungen
# Erstellen von Boxplots für ausgewählte Variablen
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.boxplot(x='grade', data=df_cleaned, ax=axes[0])
axes[0].set_title('Boxplot der Noten (grade)')

sns.boxplot(x='age', data=df_cleaned, ax=axes[1])
axes[1].set_title('Boxplot des Alters (age)')

sns.boxplot(x='absences', data=df_cleaned, ax=axes[2])
axes[2].set_title('Boxplot der Abwesenheiten (absences)')

plt.tight_layout()
plt.show()

# Das bereinigte Dataset speichern
df_cleaned.to_csv(cleaned_data_path, index=False)

print(f"Bereinigtes Dataset gespeichert unter: {cleaned_data_path}")