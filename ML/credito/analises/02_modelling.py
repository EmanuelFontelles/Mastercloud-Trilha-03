# Databricks notebook source
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

df = pd.read_parquet("/Workspace/Repos/emanuelfontelles@hotmail.com/Mastercloud-Trilha-03/ML/credito/data/treino_processado.parquet")

# COMMAND ----------

df

# COMMAND ----------

y = df["inadimplente"]
X = df.copy().drop(columns="inadimplente")

# COMMAND ----------

X

# COMMAND ----------

y

# COMMAND ----------

# MAGIC %md
# MAGIC # Treinamento do modelo

# COMMAND ----------

display(X)

# COMMAND ----------

display(X_train), display(X_test)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# Initialize MLflow
mlflow.start_run()

# COMMAND ----------

y.value_counts(normalize=True)

# COMMAND ----------

# Create a LightGBM classifier
clf = lgb.LGBMClassifier(objective='binary', class_weight='balanced')

# COMMAND ----------

clf

# COMMAND ----------

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Add more preprocessing steps as needed
    ('classifier', clf)
])

# COMMAND ----------

pipeline

# COMMAND ----------

# Log model hyperparameters and other details to MLflow
params = {
    "objective": "binary",
    "class_weight": "balanced",
}
mlflow.log_params(params)

# COMMAND ----------

# Train the model
pipeline.fit(X_train, y_train)

# Log the model to MLflow
mlflow.lightgbm.log_model(clf, "LightGBM_Model")

# COMMAND ----------

# MAGIC %md
# MAGIC # Validação do modelos

# COMMAND ----------

display(X_test.iloc[1])

# COMMAND ----------

y_test.iloc[1]

# COMMAND ----------

pipeline.predict(X_test)[1]

# COMMAND ----------

pipeline.predict(X_test)

# COMMAND ----------

from sklearn.metrics import accuracy_score

accuracy_score(y_test, pipeline.predict(X_test))

# COMMAND ----------

pipeline.predict_proba(X_test)[:, 1]

# COMMAND ----------

X_politica = X_test.copy()
X_politica['y_test'] = y_test
X_politica['predicacao'] = pipeline.predict(X_test)
X_politica['probabilidade'] = pipeline.predict_proba(X_test)[:, 1]
X_politica['score'] = 1000*pipeline.predict_proba(X_test)[:, 0]

# COMMAND ----------

display(X_politica)

# COMMAND ----------

X_politica["range"] = pd.cut(X_politica['probabilidade'], 10, labels=False)

# COMMAND ----------

display(X_politica)

# COMMAND ----------

X_politica.groupby("range")['probabilidade'].mean().plot.bar()

# COMMAND ----------

# Evaluate the model
y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(report)

# COMMAND ----------

# Calcula a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plota a matriz de confusão
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predito pelo Modelo')
plt.ylabel('Inadimplentes reais')
plt.title('Matriz de confusão')
plt.show()

# COMMAND ----------

X_test.iloc[0].to_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC # Explicabilidade do modelo

# COMMAND ----------

shap.initjs()

# COMMAND ----------

# Log metrics to MLflow
mlflow.log_metric("f1_score", f1)

# Verifique o tipo de X_test
print(type(X_test))

# Verifique a versão das bibliotecas
import shap
import lightgbm
print("SHAP version:", shap.__version__)
print("LightGBM version:", lightgbm.__version__)

# SHAP Interpretability
try:
    explainer = shap.Explainer(clf.booster_)
    shap_values = explainer(X_test)
    
    # Verifique os valores de SHAP
    print("SHAP values:", shap_values)
    
    # Log SHAP values (you can also visualize them)
    shap.summary_plot(shap_values, X_test)
except Exception as e:
    print("An error occurred:", e)

# COMMAND ----------

explainer = shap.Explainer(clf.booster_)
shap_values = explainer(X)

# Verifique os valores de SHAP
print("SHAP values:", shap_values)

# Log SHAP values (you can also visualize them)
shap.summary_plot(shap_values, X)

# COMMAND ----------

# End the MLflow run
mlflow.end_run()

# COMMAND ----------

import time
time.sleep(3600*2)
