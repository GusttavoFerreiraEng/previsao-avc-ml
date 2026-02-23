import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CARREGAMENTO E LIMPEZA DE DADOS (DATA CLEANING)
df = pd.read_csv('stroke.csv')

# Tratando valores ausentes: o BMI (IMC) é crucial, então usamos a média para não perder dados
df['bmi'] = df['bmi'].fillna(df['bmi'].mean()) 
# Para o status de fumante, preenchemos os vazios como 'Unknown' (Desconhecido)
df['smoking_status'] = df['smoking_status'].fillna('Unknown')

# 2. PRÉ-PROCESSAMENTO (FEATURE ENGINEERING)
# Transformando categorias de texto em números, pois modelos de ML só processam cálculos
le = LabelEncoder()
colunas_textos = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for col in colunas_textos:
    df[col] = le.fit_transform(df[col].astype(str))

# Separando as características (X) do alvo que queremos prever (y = stroke)
# Removemos o 'id' pois ele não ajuda na previsão (é apenas um número sequencial)
X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# 3. DIVISÃO DOS DADOS (TRAIN/TEST SPLIT)
# Separamos 20% para teste para garantir que a IA não "decore" as respostas (Overfitting)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

print(f"Dados prontos. Temos {len(X_train)} exemplos para treino e {len(X_test)} para teste.")

# 4. CRIAÇÃO DO MODELO
# Usamos class_weight='balanced' para dar mais importância aos raros casos de AVC
# Isso evita que o modelo seja "preguiçoso" e diga que ninguém terá AVC
modelo = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Validação Cruzada: Testa o modelo em 5 partes diferentes dos dados para garantir robustez
scores = cross_val_score(modelo, X, y, cv=5)
print(f"Acurácia real média (Cross-Validation): {scores.mean() * 100:.2f}%")

print("Treinando o modelo... aguarde um instante")
modelo.fit(X_train, y_train)

# 5. AVALIAÇÃO E ANÁLISE DE SENSIBILIDADE (THRESHOLD)
# Predição padrão (50% de certeza)
predicoes = modelo.predict(X_test)

# Obtendo as probabilidades para criar uma IA mais "Rigorosa"
probabilidades = modelo.predict_proba(X_test)[:, 1]

# Criamos uma predição onde 30% de chance já aciona o alerta (Foco em salvar vidas)
pred_rigorosa = (probabilidades > 0.3).astype(int)

print(f'\nTreinamento concluído. Acurácia do modelo: {accuracy_score(y_test, predicoes) * 100:.2f}%')
print('\nRelatório de Classificação:')
print(classification_report(y_test, predicoes))

# 6. VISUALIZAÇÃO DOS RESULTADOS
# Comparativo de Matrizes de Confusão: IA Normal vs IA Rigorosa
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, predicoes), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('IA Normal (Threshold 0.5)')

sns.heatmap(confusion_matrix(y_test, pred_rigorosa), annot=True, fmt='d', cmap='Reds', ax=ax[1])
ax[1].set_title('IA Rigorosa (Threshold 0.3)')

plt.show()

# Gráfico de Importância das Características (O que a IA considera mais relevante)
importances = modelo.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features, palette='viridis')
plt.title('Quais fatores a IA considera mais importantes para o AVC?')
plt.xlabel('Nível de Importância')
plt.ylabel('Características (Features)')
plt.show()

# Visualização final dos dados tratados
print(df.head())