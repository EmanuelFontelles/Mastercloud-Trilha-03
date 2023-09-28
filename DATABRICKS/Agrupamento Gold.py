# Databricks notebook source
# dbutils.fs.mount(
#   source = "wasbs://alphavantage@mastercloud.blob.core.windows.net/",
#   mount_point = "/mnt/alphavantage",
#   extra_configs = {"fs.azure.account.key.mastercloud.blob.core.windows.net": "4b0mdqbDjhMfosG6Gv5/qJf2LhkZ2q8lplCMJ21K3sIR+aLC0mPYMcgxGnHb+WuV0JOK7w5JPJT2+AStv3wftg=="})

# COMMAND ----------

# MAGIC %fs ls /mnt/alphavantage/bronze/date_reference=2023-09-28/

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

df = spark.read.json("/mnt/alphavantage/bronze")
df = df.withColumnRenamed("Time Series (Daily)", "TimeSeries")

# COMMAND ----------

from pyspark.sql.functions import col, explode
df.selectExpr("explode(map_from_entries(named_struct('date', date_reference, 'open'))) as DailyTimeSeries")\
  .select("*")

# COMMAND ----------

display(
  df
  .groupby("date_reference")
  .agg(
    F.count("ticket")
  )
  .write.format("parquet").mode("overwrite").saveAsTable("alphavantage_tickets")
)
