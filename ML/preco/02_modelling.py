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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

spark_df = spark.read.table("hive_metastore.default.price_train_gold")
df = spark_df.toPandas()

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #Modelagem

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dados de treino e teste

# COMMAND ----------

y = df["price_range"]
# X = df.drop(columns="price_range")
# X = df[['ram']]
X = df.drop(columns=["price_range", "ram"])

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# COMMAND ----------

mlflow.start_run()

# COMMAND ----------

# Define numerical features (all features in this case)
numerical_features = X.columns.tolist()

# Create a numerical transformer with StandardScaler
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)
    ])

# COMMAND ----------

clf = lgb.LGBMClassifier(objective='multiclass', class_weight='balanced')
# clf = lgb.LGBMRegressor(objective='multiclass', class_weight='balanced')

# COMMAND ----------

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', clf)
])

# COMMAND ----------

pipeline

# COMMAND ----------

pipeline.fit(X_train, y_train)

# COMMAND ----------

y_pred = pipeline.predict(X_test)

# COMMAND ----------

# antigo
print(classification_report(y_test, y_pred))

# COMMAND ----------

# novo
print(classification_report(y_test, y_pred))

# COMMAND ----------

# novo sem "ram"
print(classification_report(y_test, y_pred))

# COMMAND ----------

conf_matrix = confusion_matrix(y_test, y_pred)

# COMMAND ----------

sns.heatmap(conf_matrix, annot=True, fmt="d")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Variáveis importantes

# COMMAND ----------

feature_importances = pipeline.named_steps['classifier'].feature_importances_

# COMMAND ----------

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': numerical_features,
    'Importance': feature_importances
})

# COMMAND ----------

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# COMMAND ----------

import shap

# Create the explainer object
explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])

# Transform the test set
X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)

# Compute SHAP values
shap_values = explainer.shap_values(X_test_transformed)

# Plot
shap.summary_plot(shap_values, X_test_transformed, feature_names=numerical_features)

# COMMAND ----------


