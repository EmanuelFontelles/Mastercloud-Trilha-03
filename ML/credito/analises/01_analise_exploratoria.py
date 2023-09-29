# Databricks notebook source
from pyspark.sql import functions as F
import pandas as pd

# COMMAND ----------

df = spark.read.csv("file:/Workspace/Repos/emanuelfontelles@hotmail.com/Mastercloud-Trilha-03-Prod/ML/credito/data/treino.csv", header=True)

# COMMAND ----------

df = df.withColumn('idade', F.col('idade').cast('int'))
df = df.withColumn('salario_mensal', F.col('salario_mensal').cast('float'))
df = df.withColumn('razao_debito', F.col('razao_debito').cast('float'))
df = df.withColumn('inadimplente', F.col('inadimplente').cast('int'))
df = df.withColumn('util_linhas_inseguras', F.col('util_linhas_inseguras').cast('float'))
df = df.withColumn('vezes_passou_de_30_59_dias', F.col('vezes_passou_de_30_59_dias').cast('float'))
df = df.withColumn('numero_linhas_crdto_aberto', F.col('numero_linhas_crdto_aberto').cast('int'))
df = df.withColumn('util_linhas_inseguras', F.col('util_linhas_inseguras').cast('float'))

# COMMAND ----------

df = df.filter((F.col('idade')> 20) & (F.col('idade')< 80))

# COMMAND ----------

df_pd = df.toPandas()

# COMMAND ----------

display(df.corr("salario_mensal", "numero_linhas_crdto_aberto"))

# COMMAND ----------

df_pd.corr()

# COMMAND ----------

display(
    df
    .groupby("idade")
    .agg(F.mean('inadimplente'))
)

# COMMAND ----------

import time
time.sleep(3600*2)

# COMMAND ----------


