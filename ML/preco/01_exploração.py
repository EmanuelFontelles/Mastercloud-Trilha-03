# Databricks notebook source
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploração

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leitura do dado

# COMMAND ----------

spark_df = spark.read.table("hive_metastore.default.price_train")
df = spark_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploração e agrupamentos

# COMMAND ----------

df['price_range'].value_counts()

# COMMAND ----------

df.groupby(["price_range"])['blue'].value_counts(normalize=True)

# COMMAND ----------

df.groupby(["price_range"])['dual_sim'].value_counts(normalize=True)

# COMMAND ----------

df.groupby(["price_range"])['n_cores'].describe()

# COMMAND ----------

df.groupby(["price_range"])['battery_power'].describe()

# COMMAND ----------

df.groupby(["price_range"])['clock_speed'].describe()

# COMMAND ----------

df.groupby(["price_range", 'n_cores']).size().unstack()

# COMMAND ----------

df.groupby(["price_range", 'touch_screen'])['ram'].mean()

# COMMAND ----------

df['area'] = df['sc_h'] * df['sc_w']
spark_df = spark_df.withColumn('area', spark_df['sc_h'] * spark_df['sc_w'])

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlação

# COMMAND ----------

import seaborn as sns

# COMMAND ----------

df.corr()

# COMMAND ----------

sns.heatmap(df.corr(), cmap="viridis", )

# COMMAND ----------

from pandas_profiling import ProfileReport

# COMMAND ----------

profile = ProfileReport(df)

# COMMAND ----------

report_html = profile.to_html()
displayHTML(report_html)

# COMMAND ----------

spark_df.write.saveAsTable("hive_metastore.default.price_train_gold")

# COMMAND ----------


