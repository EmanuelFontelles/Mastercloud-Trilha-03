# Databricks notebook source
from pyspark.sql import functions as F
import pandas as pd

# COMMAND ----------

df = spark.read.csv("file:/Workspace/Repos/emanuelfontelles@hotmail.com/Mastercloud-Trilha-03/ML/credito/data/treino.csv", header=True)

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.withColumn('idade', F.col('idade').cast('int'))
df = df.withColumn('salario_mensal', F.col('salario_mensal').cast('float'))
df = df.withColumn('razao_debito', F.col('razao_debito').cast('float'))
df = df.withColumn('inadimplente', F.col('inadimplente').cast('int'))
df = df.withColumn('numero_emprestimos_imobiliarios', F.col('numero_emprestimos_imobiliarios').cast('float'))
df = df.withColumn('util_linhas_inseguras', F.col('util_linhas_inseguras').cast('float'))
df = df.withColumn('vezes_passou_de_30_59_dias', F.col('vezes_passou_de_30_59_dias').cast('float'))
df = df.withColumn('numero_de_vezes_que_passou_60_89_dias', F.col('numero_de_vezes_que_passou_60_89_dias').cast('float'))
df = df.withColumn('numero_vezes_passou_90_dias', F.col('numero_vezes_passou_90_dias').cast('float'))
df = df.withColumn('numero_linhas_crdto_aberto', F.col('numero_linhas_crdto_aberto').cast('int'))
df = df.withColumn('util_linhas_inseguras', F.col('util_linhas_inseguras').cast('float'))

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.filter((F.col('idade')> 18) & (F.col('idade')< 80))
df = df.filter((F.col('salario_mensal')> 100) & (F.col('salario_mensal')< 20000))

# COMMAND ----------

df = df.withColumn(
    "atraso",
    F.when(
        df['vezes_passou_de_30_59_dias'] + df['numero_de_vezes_que_passou_60_89_dias'] + df['numero_vezes_passou_90_dias'] > 0,
        0
    ).otherwise(1)
)

# COMMAND ----------

df.groupby("numero_emprestimos_imobiliarios", "atraso").agg(F.mean("inadimplente")).display()

# COMMAND ----------

df_pd = df.toPandas()

# COMMAND ----------

df_pd.describe()

# COMMAND ----------

df_pd['salario_mensal'].value_counts().sort_index()

# COMMAND ----------

df_pd.corr(method="spearman")

# COMMAND ----------

display(
    df
    .groupby("idade")
    .agg(F.mean('inadimplente'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Salvar nosso dado para a camada de modelagem

# COMMAND ----------

df_pd.to_parquet("/Workspace/Repos/emanuelfontelles@hotmail.com/Mastercloud-Trilha-03/ML/credito/data/treino_processado.parquet")

# COMMAND ----------

import time
time.sleep(3600*2)

# COMMAND ----------


