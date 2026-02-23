import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('stroke.csv')

df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
df['smoking_status'] = df['smoking_status'].fillna('Unknown')

le = LabelEncoder()
colunas_textos = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for col in colunas_textos:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dados prontos, Temos {len(X_train)} exemplos para treino e {len(X_test)} para teste.")

modelo = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

scores = cross_val_score(modelo, X, y, cv=5)
print(f"Acurácia real média: {scores.mean() * 100:.2f}%")

print("Treinando o modelo...aguarde um instante")
modelo.fit(X_train, y_train)

predicoes = modelo.predict(X_test)

probabilidades = modelo.predict_proba(X_test)[:, 1]

predicoes_rigorosas = (probabilidades > 0.3).astype(int)

acuracia = accuracy_score(y_test, predicoes)
print(f'\nTreinamento concluido')
print(f' Acurácia do modelo: {acuracia * 100:.2f}%')

print('\nRelatório de Classificação (Classification Report):')
print(classification_report(y_test, predicoes))

cm = confusion_matrix(y_test, predicoes)

probs = modelo.predict_proba(X_test)[:, 1]

pred_normal = (probs > 0.5).astype(int)

pred_rigorosa = (probs > 0.3).astype(int)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, pred_normal), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('IA Normal (Pode ser preguiçosa)')

sns.heatmap(confusion_matrix(y_test, pred_rigorosa), annot=True, fmt='d', cmap='Reds', ax=ax[1])
ax[1].set_title('IA Rigorosa (Detetive de AVC)')

plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsão do Modelo')
plt.ylabel('Realidade (Paciente)')
plt.title('Matriz de Confusão: Acertos vs Erros')
plt.show()

importances = modelo.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title('Quais fatores a IA considera mais importantes para o AVC?')
plt.xlabel('Nível de Importância')
plt.ylabel('Características (Features)')
plt.show()

df.head()