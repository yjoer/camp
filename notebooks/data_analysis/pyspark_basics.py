# %%
import os
import sys

os.environ["SPARK_LOCAL_IP"] = "0.0.0.0"

if sys.platform == "win32":
    os.environ["PATH"] += os.pathsep + os.getenv("HADOOP_HOME", "") + "/bin"
else:
    os.environ["LD_LIBRARY_PATH"] = os.getenv("HADOOP_HOME", "") + "/lib/native"

# %%
import tempfile
from datetime import date
from datetime import datetime

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.types import DateType
from pyspark.sql.types import FloatType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
from pyspark.sql.types import TimestampType

# %%
spark = (
    SparkSession.builder.config("spark.ui.showConsoleProgress", False)
    .config("spark.sql.execution.arrow.pyspark.enabled", True)
    .getOrCreate()
)

# %% [markdown]
# ## DataFrame Creation

# %% [markdown]
# Create a PySpark DataFrame from a list of rows.

# %%
rows = [
    Row(a=1, b=1.1, c="a", d=date(2023, 1, 1), e=datetime(2023, 1, 1, 12, 0, 0)),
    Row(a=2, b=2.2, c="b", d=date(2024, 1, 1), e=datetime(2024, 1, 1, 12, 0, 0)),
    Row(a=3, b=3.3, c="c", d=date(2025, 1, 1), e=datetime(2025, 1, 1, 12, 0, 0)),
]

df = spark.createDataFrame(rows)
df

# %% [markdown]
# Create a PySpark DataFrame with an explicit schema.

# %%
schema = "a int, b float, c string, d date, e timestamp"
df = spark.createDataFrame(rows, schema=schema)
df

# %%
schema = StructType(
    [
        StructField("a", IntegerType()),
        StructField("b", FloatType()),
        StructField("c", StringType()),
        StructField("d", DateType()),
        StructField("e", TimestampType()),
    ]
)

df = spark.createDataFrame(rows, schema=schema)
df

# %% [markdown]
# Create a PySpark DataFrame from a Pandas DataFrame.

# %%
df_pd = pd.DataFrame(
    {
        "a": [1, 2, 3],
        "b": [1.1, 1.2, 1.3],
        "c": ["a", "b", "c"],
        "d": [date(2023, 1, 1), date(2024, 1, 1), date(2025, 1, 1)],
        "e": [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2025, 1, 1, 12, 0, 0),
        ],
    }
)

df = spark.createDataFrame(df_pd)
df

# %%
df.show()
df.printSchema()

# %% [markdown]
# ## Viewing Data

# %% [markdown]
# Display the top rows of a DataFrame.

# %%
df.show(1)

# %% [markdown]
# Enable eager evaluation of the DataFrame in notebooks.

# %%
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
spark.conf.set("spark.sql.repl.eagerEval.maxNumRows", 3)

df

# %% [markdown]
# Show the rows vertically when they are too long to show horizontally.

# %%
df.show(1, vertical=True)

# %% [markdown]
# List the column names.

# %%
df.columns

# %% [markdown]
# Show the summary of the DataFrame.

# %%
df.select("a", "b", "c").describe().show()

# %% [markdown]
# Collect the distributed data to the driver side as the local data in Python.

# %%
df.collect()

# %% [markdown]
# If the data residing in executors is too large to fit on the driver side, we can take a subset of the data.

# %%
df.take(1)

# %%
df.tail(1)

# %% [markdown]
# Convert the PySpark DataFrame back to the Pandas DataFrame.

# %%
df.toPandas()

# %% [markdown]
# ## Selecting and Accessing Data

# %% [markdown]
# Selecting a column does not trigger a computation because PySpark DataFrame is lazily evaluated.

# %%
df.a

# %% [markdown]
# Most column-wise operations return a column instance.

# %%
type(df.c) is type(F.upper(df.c)) is type(df.c.isNull())

# %% [markdown]
# A column instance can be used to select columns from a DataFrame.

# %%
df.select(df.c).show()

# %% [markdown]
# Assign a new column instance.

# %%
df.withColumn("a_upper", F.upper(df.c)).show()

# %% [markdown]
# Select a subset of rows.

# %%
df.filter(df.a == 1).show()


# %% [markdown]
# ## Apply a Function

# %% [markdown]
# Pandas UDF


# %%
@F.pandas_udf("long")  # type: ignore
def plus_one(series: pd.Series) -> pd.Series:
    return series + 1


df.select(plus_one(df.a)).show()


# %% [markdown]
# Use Pandas APIs without any restrictions on the result length. We can yield either none or many rows for each input row.


# %%
def filter_one(iterator):
    for df_pd in iterator:
        yield df_pd[df_pd.a == 1]


df.mapInPandas(filter_one, schema=df.schema).show()

# %% [markdown]
# ## Grouping Data

# %%
df = spark.createDataFrame(
    [
        ("a", "carrot", 1, 10),
        ("b", "carrot", 4, 40),
        ("c", "carrot", 7, 70),
        ("b", "banana", 2, 20),
        ("a", "banana", 5, 50),
        ("c", "banana", 8, 80),
        ("b", "grape", 3, 30),
        ("c", "grape", 6, 60),
        ("a", "grape", 9, 90),
    ],
    schema=["basket", "fruit", "v1", "v2"],
)

# %% [markdown]
# Compute the averages for groups.

# %%
df.groupby("basket").avg().show()


# %% [markdown]
# Apply a Python function against each group using the Pandas API.


# %%
def minus_mean(df_pd):
    return df_pd.assign(v1=df_pd.v1 - df_pd.v1.mean())


df.groupby("basket").applyInPandas(minus_mean, schema=df.schema).show()

# %% [markdown]
# Co-group and apply a function.

# %%
df1 = spark.createDataFrame(
    [
        (20250101, 1, 1.0),
        (20250101, 2, 2.0),
        (20250102, 1, 3.0),
        (20250102, 2, 4.0),
    ],
    schema=["time", "id", "v1"],
)

df2 = spark.createDataFrame(
    [(20250101, 1, "x"), (20250101, 2, "y")],
    schema=["time", "id", "v2"],
)


def merge_ordered(left, right):
    return pd.merge_ordered(left, right)


df1.groupby("id").cogroup(df2.groupby("id")).applyInPandas(
    merge_ordered,
    schema="time int, id int, v1 double, v2 string",
).show()

# %% [markdown]
# ## Input/Output

# %% [markdown]
# CSV

# %%
with tempfile.TemporaryDirectory() as d:
    df.write.csv(d, mode="overwrite", header=True)
    spark.read.csv(d, header=True).show()

# %% [markdown]
# Parquet

# %%
with tempfile.TemporaryDirectory() as d:
    df.write.parquet(d, mode="overwrite")
    spark.read.parquet(d, header=True).show()

# %% [markdown]
# ORC

# %%
with tempfile.TemporaryDirectory() as d:
    df.write.orc(d, mode="overwrite")
    spark.read.orc(d).show()

# %% [markdown]
# ## SQL

# %% [markdown]
# Register the DataFrame as a table and run an SQL using the same execution engine.

# %%
df.createOrReplaceTempView("table")
spark.sql("SELECT count(*) FROM table").show()

# %% [markdown]
# Register a UDF and invoke it in an SQL query.

# %%
spark.udf.register("plus_one", plus_one)
spark.sql("SELECT plus_one(v1) FROM table").show()

# %% [markdown]
# Mix and use SQL expressions as PySpark columns.

# %%
df.selectExpr("plus_one(v1)").show()
df.select(F.expr("count(*)") > 0).show()

# %%
