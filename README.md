# 🧠 Previsão de Risco de AVC - Machine Learning

Este projeto utiliza algoritmos de Machine Learning para prever a probabilidade de um paciente sofrer um AVC (Acidente Vascular Cerebral) com base em fatores clínicos e demográficos. O modelo foi desenvolvido focando na redução de falsos negativos, priorizando a segurança em diagnósticos de saúde.

## 🚀 Diferenciais Técnicos (Análise de Engenharia)

Diferente de modelos básicos, este projeto aborda desafios reais de Ciência de Dados que aprendi no ciclo de vida de ML da AWS:

* **Tratamento de Dados Desbalanceados:** Como a base de dados possui poucos casos positivos de AVC (~2%), utilizei o parâmetro `class_weight='balanced'` no RandomForest para evitar que a IA fosse "preguiçosa" e ignorasse os casos de risco.
* **Ajuste de Sensibilidade (Thresholding):** Configurei o limiar de decisão (threshold) em **0.3**. Isso torna o modelo mais rigoroso: ele alerta sobre o risco mesmo quando a probabilidade não é absoluta, priorizando o "Recall" sobre a acurácia bruta.
* **Validação Robusta:** Utilizei **Cross-Validation (5-folds)** para garantir que a acurácia de ~98% fosse consistente em diferentes partes do dataset, evitando o Overfitting (quando a IA apenas decora os dados).



## 📊 Fatores Mais Importantes

A análise de importância das características (Feature Importance) revelou que os principais preditores para este modelo são:
1.  **Nível Médio de Glicose**
2.  **IMC (BMI)**
3.  **Idade (Age)**

## 🛠️ Tecnologias Utilizadas

* **Python 3.x**
* **Pandas:** Manipulação e limpeza de dados.
* **Scikit-Learn:** Criação do modelo, treino e avaliação.
* **Seaborn & Matplotlib:** Visualização de matrizes de confusão e gráficos de importância.

## 📈 Como rodar o projeto

1. Clone o repositório:
   ```bash
   git clone [https://github.com/GusttavoFerreiraEng/previsao-avc-ml.git](https://github.com/GusttavoFerreiraEng/previsao-avc-ml.git)