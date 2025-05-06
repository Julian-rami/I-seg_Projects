# Databricks notebook source
# MAGIC %md
# MAGIC # Big Data Tools Final Project
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Authors:
# MAGIC - **Rohan Taneja**
# MAGIC - **Julian Subagiyo**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Project Overview:
# MAGIC This notebook serves as the final project submission for the Big Data Tools course. It includes the analysis, insights, and implementation details related to the project topic of review score prediction.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler, Bucketizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression ,GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator ,BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import functions as F



# COMMAND ----------

# MAGIC %md
# MAGIC Initializing Spark Session

# COMMAND ----------

spark = SparkSession.builder \
    .master("local") \
    .appName("BigDataProject") \
    .getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC Imports

# COMMAND ----------

def load_csv(file_name):
    """Helper function to load a CSV file into a Spark DataFrame."""
    return spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(file_name)

# Base directory for data files
data_dir = r"C:\Users\omgit\Desktop\Big Data Proj\Data-20250203"

# Load datasets
order_items = load_csv(f"{data_dir}\order_items.csv")
products = load_csv(f"{data_dir}\products.csv")
order_payments = load_csv(f"{data_dir}\order_payments.csv")
order_reviews = load_csv(f"{data_dir}\order_reviews.csv")
orders = load_csv(f"{data_dir}\orders.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC Function that checks for NULLS or NA values in a given dataset:

# COMMAND ----------

# Function to count missing values
def count_missing_values(df):
    return df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])

# Function to count 'NA' values
def count_na_values(df):
    return df.select([count(when(col(c) == 'NA', c)).alias(c) for c in df.columns])

# Display missing and 'NA' values count for each dataset
for df, name in zip([order_items, products, order_payments, order_reviews], 
                    ["order_items", "products", "order_payments", "order_reviews",'orders']):
    print(f"Missing values in {name}:")
    display(count_missing_values(df))
    print(f"'NA' values in {name}:")
    display(count_na_values(df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Order Items and Products Table

# COMMAND ----------

# MAGIC %md
# MAGIC We first partition by order_id to sort the data and use aggregate functions to create total_price_of_order and total_shipping_of_order

# COMMAND ----------

window_spec = Window.partitionBy("order_id")
order_items = order_items.withColumn("total_items_in_order", count("order_item_id").over(window_spec))

total_costs = order_items.groupBy("order_id").agg(
    sum(col("price")).alias("total_price_of_order"),
    sum(col("shipping_cost")).alias("total_shipping_of_order")
    )

total_costs.orderBy("order_id").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Join the order_items DataFrame with the total_costs DataFrame on the "order_id" column
# MAGIC . Drop the columns "order_item_id", "price", and "shipping_cost" from the order_items DataFrame

# COMMAND ----------

order_items = order_items.join(total_costs, "order_id", how='left')

order_items = order_items.drop("order_item_id", "price", "shipping_cost")

# COMMAND ----------

# MAGIC %md
# MAGIC Sort by order_id for next aggregate functions

# COMMAND ----------

order_items = order_items.orderBy("order_id")

# COMMAND ----------

# MAGIC %md
# MAGIC We choose to join product table here since it will allow us to create new features easier (left join because it's products has less rows than order_items)

# COMMAND ----------

order_items_products = order_items.join(products,"product_id","left")

# COMMAND ----------

# MAGIC %md
# MAGIC Dropping NA values from the join

# COMMAND ----------

order_items_products = order_items_products.dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC We identify upto 70 different product categories in this table

# COMMAND ----------

distinct_values = order_items_products.select("product_category_name").distinct()
distinct_values.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Creating our first feature 'product_volume' by using dimensions and basic math

# COMMAND ----------

order_items_products = order_items_products.withColumn(
    'product_volume',
    col('product_length_cm') * col('product_height_cm') * col('product_width_cm')
)

# COMMAND ----------

# MAGIC %md
# MAGIC Aggregation of order-level data:
# MAGIC
# MAGIC - average_product_desc_length: Average length of product descriptions in an order
# MAGIC - average_product_name_length: Average length of product names in an order
# MAGIC - total_items_in_order: Total number of items in an order
# MAGIC - total_price_of_order: Total price of all items in an order
# MAGIC - total_shipping_of_order: Total shipping cost for an order
# MAGIC - sum_of_all_products_weight: Sum of the weights of all products in an order
# MAGIC - categories: Unique categories of products stored in an array
# MAGIC - total_photos_in_order: Total number of photos associated with products in an order
# MAGIC - total_product_volume: Sum of product volumes in an order
# MAGIC

# COMMAND ----------

order_items_grouped = order_items_products.groupBy("order_id").agg(
    
    avg("product_description_lenght").alias("average_product_desc_length"),
    avg("product_name_lenght").alias("average_product_name_length"),  
    round(F.first("total_items_in_order"),2).alias("total_items_in_order"),       
    round(F.first("total_price_of_order"),2).alias("total_price_of_order"),
    round(F.first("total_shipping_of_order"),2).alias("total_shipping_of_order"),
    sum('product_weight_g').alias('sum_of_all_products_weight'),
    F.first("product_category_name").alias("category"),
    sum('product_photos_qty').alias('total_photos_in_order'),
    sum('product_volume').alias('total_product_volume')        
)

# COMMAND ----------

# MAGIC %md
# MAGIC Replacing our original dataset with the new one

# COMMAND ----------

order_items_products = order_items_grouped

# COMMAND ----------

# MAGIC %md
# MAGIC Overview of certain features so we can bin them

# COMMAND ----------

order_items_products.select("average_product_desc_length").summary().show()
order_items_products.select("average_product_name_length").summary().show()
order_items_products.select("sum_of_all_products_weight").summary().show()
order_items_products.select("total_photos_in_order").summary().show()
order_items_products.select("total_product_volume").summary().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Binning

# COMMAND ----------

bucketBorders = [3.0, 350.0, 600.0, 985.0, 4000.0]
order_items_products = Bucketizer(inputCol="average_product_desc_length", outputCol="Buckets_descr",splits=bucketBorders).transform(order_items_products)
bucketBorders_name = [3.0, 50.0, 100.0]
order_items_products = Bucketizer(inputCol="average_product_name_length", outputCol="Buckets_name",splits=bucketBorders_name).transform(order_items_products)
bucketBorders_weight = [0.0, 300.0, 800.0, 2000.0, 150000.0]
order_items_products = Bucketizer(inputCol="sum_of_all_products_weight", outputCol="Buckets_weight",splits=bucketBorders_weight).transform(order_items_products)
bucketBorders_volume = [0.0, 3000.0, 8000.0, 20000.0, 900000.0]
order_items_products = Bucketizer(inputCol="total_product_volume", outputCol="Buckets_volume",splits=bucketBorders_volume).transform(order_items_products)

# COMMAND ----------

# MAGIC %md
# MAGIC and subsequent dropping of existing variables

# COMMAND ----------

columns_to_drop = [
    "average_product_desc_length", 
    "average_product_name_length", 
    "sum_of_all_products_weight", 
    "total_product_volume"
]

order_items_products = order_items_products.drop(*columns_to_drop)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Orders table

# COMMAND ----------

# MAGIC %md
# MAGIC Checking for nulls

# COMMAND ----------

count_missing_values(orders).show()
count_na_values(orders).show()

# COMMAND ----------

# MAGIC %md
# MAGIC We chose to exclude any orders that were not marked as "delivered," as we believed the reviews associated with them stemmed from an inconsistent process and would not provide meaningful insights into the reviews.

# COMMAND ----------

orders = orders.where(col("order_status")== "delivered").select("*")
orders = orders.drop("order_status")

# COMMAND ----------

# MAGIC %md
# MAGIC Defining a new variable, delivery time, measured in days.

# COMMAND ----------

orders = orders.withColumn(
    "delivery_time",
    when(
        col("order_purchase_timestamp").isNotNull() & col("order_delivered_customer_date").isNotNull(),
        round(((unix_timestamp(col("order_delivered_customer_date")) - unix_timestamp(col("order_purchase_timestamp"))) / 86400).cast("double"), 2)
    ).otherwise(-1)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Checking for nulls after we create the new variable.

# COMMAND ----------

count_missing_values(orders).show()
count_na_values(orders).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Dropping nulls

# COMMAND ----------

orders = orders.dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC Over here we do the following:
# MAGIC
# MAGIC 1. Convert Dates to Timestamps:
# MAGIC    - The `order_delivered_customer_date` and `order_estimated_delivery_date` columns are converted to timestamp format for further time-based calculations.
# MAGIC
# MAGIC 2. Add `delivered_on_time` Column:
# MAGIC    - Creates a new boolean column that checks if the order was delivered on or before the estimated delivery date.
# MAGIC
# MAGIC 3. Add `estimated_vs_actual_delay_days` Column:
# MAGIC    - Computes the difference (in days) between the actual delivery date and the estimated delivery date.
# MAGIC    - Positive values indicate delays, while negative values indicate early delivery.
# MAGIC
# MAGIC 4. Add `delivered_early` Column:
# MAGIC    - Creates a boolean column indicating if the order was delivered earlier than the estimated delivery date.
# MAGIC
# MAGIC 5. Add `approval_time` Column:
# MAGIC    - Calculates the time (in days) it took for the order to be approved after the purchase timestamp.
# MAGIC    - If any of the timestamps are missing, assigns a default value of `-1`.
# MAGIC
# MAGIC 6. Add `delay_status` Column:
# MAGIC    - Creates a boolean column indicating whether the order was delayed (actual delivery date is later than the estimated delivery date).

# COMMAND ----------

# Convert date columns to timestamp format
orders = orders.withColumn("order_delivered_customer_date", to_timestamp(col("order_delivered_customer_date"))) \
               .withColumn("order_estimated_delivery_date", to_timestamp(col("order_estimated_delivery_date")))

# Add calculated columns
orders = orders \
    .withColumn(
        "delivered_on_time",
        when(col("order_delivered_customer_date") <= col("order_estimated_delivery_date"), True).otherwise(False)
    ) \
    .withColumn(
        "estimated_vs_actual_delay_days",
        round((unix_timestamp(col("order_delivered_customer_date")) - unix_timestamp(col("order_estimated_delivery_date"))) / 86400, 2)
    ) \
    .withColumn(
        "delivered_early",
        when(col("order_delivered_customer_date") < col("order_estimated_delivery_date"), True).otherwise(False)
    ) \
    .withColumn(
        "approval_time",
        when(
            col("order_purchase_timestamp").isNotNull() & col("order_approved_at").isNotNull(),
            round((unix_timestamp(col("order_approved_at")) - unix_timestamp(col("order_purchase_timestamp"))) / 86400, 2)
        ).otherwise(-1)
    ) \
    .withColumn(
        "delay_status",
        when(unix_timestamp(col("order_delivered_customer_date")) > unix_timestamp(col("order_estimated_delivery_date")), True).otherwise(False)
    )


# COMMAND ----------

# MAGIC %md
# MAGIC Dropping nulls again

# COMMAND ----------

orders= orders.dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Order Reviews Table

# COMMAND ----------

# MAGIC %md
# MAGIC Checking for nulls

# COMMAND ----------

count_na_values(order_reviews).show()
count_missing_values(order_reviews).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Dropping NAs and creating a new variable called review_response_time.

# COMMAND ----------

order_reviews = order_reviews.dropna()

order_reviews = order_reviews.withColumn(
    "review_response_time",
    when(
        col("review_creation_date").isNotNull() & col("review_answer_timestamp").isNotNull(),
        ((unix_timestamp(col("review_answer_timestamp")) - unix_timestamp(col("review_creation_date"))) / 86400).cast("int")
    ).otherwise(-1)
).drop("review_creation_date")

# COMMAND ----------

# MAGIC %md
# MAGIC Some order's have more than one review.

# COMMAND ----------

display(order_reviews.groupBy(col("order_id")).count().filter(F.col("count") > 1))
order_id_double = order_reviews.groupBy(col("order_id")).count().filter(F.col("count") > 1)

# COMMAND ----------

# MAGIC %md
# MAGIC Some orders don't have reviews.

# COMMAND ----------

order_reviews.count()

# COMMAND ----------

order_reviews.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC We decide to take the latest reviews possible of each order , which is why we filter by the highest review response time. This leaves us only with 1 review per order_id and it is the latest review.

# COMMAND ----------

# Define a window partitioned by order_id and ordered by review_response_time descending
window_spec = Window.partitionBy("order_id").orderBy(col("review_answer_timestamp").desc())

# Add a row_number column to identify the top row for each order_id
order_reviews_filtered = order_reviews.withColumn(
    "row_num", row_number().over(window_spec)
).filter(col("row_num") == 1).drop("row_num","review_answer_timestamp")

# Verify the result
#display(order_reviews_filtered)
order_reviews_filtered.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Order Payments Table

# COMMAND ----------

# MAGIC %md
# MAGIC Check for NA values.

# COMMAND ----------

count_missing_values(order_payments).show()
count_na_values(order_payments).show()

# COMMAND ----------

# MAGIC %md
# MAGIC This code block creates several features to analyze payment behavior and voucher usage in orders:
# MAGIC
# MAGIC - `if_voucher_used`: Indicates whether a voucher was used in the payment for an order. Values are 1 (voucher used) or 0 (no voucher used).
# MAGIC - `voucher_count`: Represents the total number of voucher payments made for an order.
# MAGIC - `total_voucher_payment`: Calculates the total monetary value of payments made using vouchers for an order.
# MAGIC - `total_non_voucher_payment`: Calculates the total monetary value of payments made without vouchers for an order.
# MAGIC - `max_payment_sequential`: Identifies the highest sequential payment number for an order, reflecting the order of payments.
# MAGIC - `max_payment_installments`: Calculates the total number of installments for an order, helping identify installment-based payments.
# MAGIC - `primary_payment_type`: Determines the primary payment method for an order, based on the payment type with the first payment in the sequence (`payment_sequential = 1`).
# MAGIC - `order_payments_clean`: Consolidates all the above features into a clean and structured DataFrame, providing comprehensive payment-related information for each order.
# MAGIC
# MAGIC These features are useful for understanding customer payment preferences, analyzing voucher usage, and gaining insights into revenue patterns.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col, when, max, sum, round

# Step 1: Create "voucher_used" column to identify if a voucher was used in a payment
order_payments = order_payments.withColumn(
    "voucher_used",
    when(col("payment_type") == "voucher", 1).otherwise(0)
)

# Step 2: Aggregate voucher-related metrics
voucher_used = order_payments.groupBy("order_id").agg(
    max("voucher_used").alias("if_voucher_used")
)

voucher_count_df = order_payments.groupBy("order_id").agg(
    sum("voucher_used").alias("voucher_count")
)

sum_voucher_payments = order_payments.where(col("payment_type") == "voucher").groupBy("order_id").agg(
    round(sum("payment_value"), 2).alias("total_voucher_payment")
)

# Step 3: Aggregate non-voucher-related metrics
sum_non_voucher_payments = order_payments.where(col("payment_type") != "voucher").groupBy("order_id").agg(
    round(sum("payment_value"), 2).alias("total_non_voucher_payment")
)

# Step 4: Calculate maximum sequential and installment payments
max_payment_sequential = order_payments.groupBy("order_id").agg(
    max("payment_sequential").alias("max_payment_sequential")
)

max_payment_installments = order_payments.groupBy("order_id").agg(
    sum("payment_installments").alias("max_payment_installments")
)

# Step 5: Identify primary payment types
primary_payment_types = order_payments.where(col("payment_sequential") == 1).select(
    col("order_id"), col("payment_type").alias("primary_payment_type")
)

# Step 6: Combine all metrics into a clean DataFrame
order_payments_clean = voucher_used.join(voucher_count_df, on="order_id", how="inner") \
    .join(max_payment_sequential, on="order_id", how="inner") \
    .join(sum_voucher_payments, on="order_id", how="left") \
    .join(sum_non_voucher_payments, on="order_id", how="left") \
    .join(primary_payment_types, on="order_id", how="left") \
    .join(max_payment_installments, on="order_id", how="inner") \
    .filter(col("primary_payment_type").isNotNull())

# COMMAND ----------

# MAGIC %md
# MAGIC Counting missing values

# COMMAND ----------

count_missing_values(order_payments_clean).show()
count_na_values(order_payments_clean).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Filling missing values since some people paid with only voucher or only non-voucher methods.

# COMMAND ----------

order_payments_clean_nonulls = order_payments_clean.fillna({"total_voucher_payment": 0,"total_non_voucher_payment": 0})

# COMMAND ----------

# MAGIC %md
# MAGIC Checking number of rows

# COMMAND ----------

order_payments_clean_nonulls.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creation of basetable

# COMMAND ----------

# MAGIC %md
# MAGIC Bringing it all together.

# COMMAND ----------

from pyspark.sql.functions import col, when



basetable = (
    orders
    .join(order_payments_clean_nonulls, on="order_id", how="inner")
    .join(order_items_products, on="order_id", how="inner")
    .join(order_reviews_filtered, on="order_id", how="inner")
)






# COMMAND ----------

basetable.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Existing code for text categories , we don't use it since we already have 

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder

payment_type_indexer = StringIndexer(inputCol="primary_payment_type", outputCol="primary_payment_type_index")


payment_type_encoder = OneHotEncoder(inputCol="primary_payment_type_index", outputCol="primary_payment_type_vector")


categories_indexer = StringIndexer(inputCol="category", outputCol="categories_index")


categories_encoder = OneHotEncoder(inputCol="categories_index", outputCol="categories_vector")

pipeline = Pipeline(stages=[payment_type_indexer, payment_type_encoder, categories_indexer, categories_encoder])


pipeline_model = pipeline.fit(basetable)
vectorized_df = pipeline_model.transform(basetable).drop('primary_payment_type','primary_payment_type_index','category','categories_index')

# Show the result
vectorized_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Storing in another basetable for redundancy.

# COMMAND ----------

basetable2 = basetable

# COMMAND ----------

basetable = vectorized_df

# COMMAND ----------

basetable.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC We create two columns for our two models. One will be based on binary classification and the other will be based on multiclass classification. In the case of multiclass classification, we bin review scores into bins of (1,2),(3,4),(5) and 0,1,2 respectively. For binary we bin them into (1,2,3) = 0 (also can be 'bad') , (4,5) = 1 (also can be 'good')

# COMMAND ----------

basetable = basetable.withColumn(
    "multiclass_label", 
    F.when((F.col("review_score").isin(1, 2)), 0)
     .when((F.col("review_score").isin(3, 4)), 1)
     .when(F.col("review_score") == 5, 2)
)


basetable = basetable.withColumn(
    "binary_label", 
    F.when((F.col("review_score").isin(1, 2,3)), 0)
     .when((F.col("review_score").isin( 4,5)), 1)
     
)

# COMMAND ----------

# MAGIC %md
# MAGIC We check the distribution.

# COMMAND ----------

basetable.groupBy("binary_label").count().show()
basetable.groupBy("multiclass_label").count().show()


# COMMAND ----------

basetable = basetable.dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelling

# COMMAND ----------

# MAGIC %md
# MAGIC We drop irrelevant rows from each basetable.

# COMMAND ----------

basetable_binary = basetable.drop(
    "order_id", 
    "order_purchase_timestamp", 
    "order_approved_at", 
    "order_delivered_carrier_date", 
    "order_delivered_customer_date", 
    "order_estimated_delivery_date", 
    "customer_id", 
    "review_id", 
    'categories_index',
    'categories',
    'primary_payment_type_index',
    'order_count',
    'review_score',
    'multiclass_label',
    'review_response_time'
    )

basetable_multiclass = basetable.drop(
    "order_id", 
    "order_purchase_timestamp", 
    "order_approved_at", 
    "order_delivered_carrier_date", 
    "order_delivered_customer_date", 
    "order_estimated_delivery_date", 
    "customer_id", 
    "review_id", 
    'categories_index',
    'categories',
    'primary_payment_type_index',
    'order_count',
    'review_score',
    'binary_label',
    'review_response_time'
    )



# COMMAND ----------

# MAGIC %md
# MAGIC We will use RFormula to convert the basetables into a format suitable for machine learning in the spark environment.

# COMMAND ----------

from pyspark.ml.feature import RFormula

r_form_binary = RFormula(formula="binary_label ~ .")
r_form_df_binary = r_form_binary.fit(basetable_binary).transform(basetable_binary).select("features", "label")

r_form_multiclass = RFormula(formula="multiclass_label ~ .")
r_form_df_multiclass = r_form_multiclass.fit(basetable_multiclass).transform(basetable_multiclass).select("features", "label")


# COMMAND ----------

# MAGIC %md
# MAGIC Train test split.

# COMMAND ----------

#Create a train and test set with a 70% train, 30% test split
train_b, test_b = r_form_df_binary.randomSplit([0.7, 0.3],seed=123)

train_m, test_m = r_form_df_multiclass.randomSplit([0.7, 0.3],seed=123)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Our Multiclass Classification Model

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
 
# Initialize the Random Forest Classifier
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=50,  # Number of trees
    maxDepth=5,   # Maximum depth of trees
    maxBins=16     # Number of bins used for splitting
)
 
# Train the model
rf_model = rf.fit(train_m)
 
# Make predictions on the test set
predictions = rf_model.transform(test_m)
 
 
multi_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
 
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)
 
# Compute metrics
accuracy = multi_evaluator.evaluate(predictions)
f1_score = f1_evaluator.evaluate(predictions)

 
# Print results
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")

 
# Print model parameters
print("Model Parameters:")
print(f"  numTrees: {rf_model.getNumTrees}")
print(f"  maxDepth: {rf_model.getMaxDepth()}")
print(f"  maxBins: {rf_model.getMaxBins()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Our Binary Classification Model

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
 

gbt = GBTClassifier(
    labelCol="label",
    featuresCol="features",
    maxIter=50  # Number of boosting iterations
)
 

gbt_model = gbt.fit(train_b)
 
gbt_pred = gbt_model.transform(test_b)
 

ovr_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
 
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
 
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)
 
ovr_accuracy = ovr_evaluator.evaluate(gbt_pred)
f1_score = f1_evaluator.evaluate(gbt_pred)
auc_roc = binary_evaluator.evaluate(gbt_pred)
 
print(f"GBTClassifier Test Accuracy: {ovr_accuracy}")
print(f"F1 Score: {f1_score}")
print(f"AUC-ROC: {auc_roc}")

# COMMAND ----------

predictions = gbt_pred.orderBy(col("probability").asc())
 
from pyspark.sql.window import Window
from pyspark.sql.functions import ntile
 
# Assign percentile rank (1-10)
predictions = predictions.withColumn("decile", ntile(10).over(Window.orderBy(col("probability").asc())))
 
# Get total number of positive cases
total_positives = predictions.filter(col("label") == 1).count()
total_count = predictions.count()
baseline_rate = total_positives / total_count  # Random model's response rate
 
# Compute lift per decile
lift_df = predictions.groupBy("decile").agg(
    (F.sum("label") / F.count("label") / baseline_rate).alias("lift")  # Lift = Response Rate / Baseline Rate
)
 
# Sort by decile
lift_df = lift_df.orderBy("decile")
 

# COMMAND ----------

import matplotlib.pyplot as plt
random_model = [1.0] * 10 
x = range(1, 11)  # Deciles
 
plt.plot(x, lift_df.toPandas()["lift"], marker="o", label="Model Lift", color="blue")
plt.plot(x, random_model, linestyle="--", label="Random Model (Baseline)", color="red")
 
plt.xlabel("Deciles")
plt.ylabel("Lift")
plt.title("Lift Chart")
plt.legend()
plt.grid()
plt.show()

# COMMAND ----------

import random_model

# COMMAND ----------

# Compute Confusion Matrix
confusion_matrix = predictions.groupBy("label", "prediction").count()
tp = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
fp = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
fn = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()
tn = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
 
# # Compute True Positive Rate (TPR) and False Positive Rate (FPR)
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity, Recall
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Type I Error
 
print(f"True Positive Rate (TPR): {tpr}")
print(f"False Positive Rate (FPR): {fpr}")
confusion_matrix.show()

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
# Manually creating the confusion matrix
conf_matrix = pd.DataFrame(
    [[728, 2386],  # Row: True Negative (0) / False Positive (0 → 1)
     [237, 11200]], # Row: False Negative (1 → 0) / True Positive (1 → 1)
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)
 
# Plot using Seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Importance extraction from each model

# COMMAND ----------

import pandas as pd

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))
  



# COMMAND ----------

# MAGIC %md
# MAGIC We put the features in a pandas dataframe and then plot them.

# COMMAND ----------

features_df = pd.DataFrame(ExtractFeatureImp(rf_model.featureImportances, train_m, "features"))
features_df_binary = pd.DataFrame(ExtractFeatureImp(gbt_model.featureImportances, train_b, "features"))

# COMMAND ----------

# Display the top 10 features by importance
plt.figure(figsize=(10, 6))
sns.barplot(x='score', y='name', data=features_df.head(10), palette='viridis')

# Adding labels and title
plt.title('Top 10 Features by Importance according to Multiclass', fontsize=16)
plt.xlabel('Feature Importance Score', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)

# Display the plot
plt.show()


# COMMAND ----------

# Display the top 10 features by importance
plt.figure(figsize=(10, 6))
sns.barplot(x='score', y='name', data=features_df_binary.head(10), palette='viridis')

# Adding labels and title
plt.title('Top 10 Features by Importance according to Binary', fontsize=16)
plt.xlabel('Feature Importance Score', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)

# Display the plot
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Holdout Data

# COMMAND ----------

# MAGIC %md
# MAGIC Now we do the same transformations to the holdout data.

# COMMAND ----------

data_dir_holdout = r"C:/Users/omgit/Desktop/Big Data Proj/Holdout data-20250211"

h_order_items = load_csv(f"{data_dir_holdout}/test_order_items.csv")
h_products = load_csv(f"{data_dir_holdout}/test_products.csv")
h_order_payments = load_csv(f"{data_dir_holdout}/test_order_payments.csv")

h_orders = load_csv(f"{data_dir_holdout}/test_orders.csv")

# COMMAND ----------

window_spec = Window.partitionBy("order_id")
h_order_items = h_order_items.withColumn("total_items_in_order", count("order_item_id").over(window_spec))

total_costs = h_order_items.groupBy("order_id").agg(
    sum(col("price")).alias("total_price_of_order"),
    sum(col("shipping_cost")).alias("total_shipping_of_order")
    )

h_order_items = h_order_items.join(total_costs, "order_id", how='left')

h_order_items = h_order_items.drop("order_item_id", "price", "shipping_cost")

h_order_items_products = h_order_items.join(h_products,"product_id","left")

# COMMAND ----------

h_order_items_products = h_order_items_products.dropna()

# COMMAND ----------

h_order_items_products = h_order_items_products.withColumn(
    'product_volume',
    col('product_length_cm') * col('product_height_cm') * col('product_width_cm')
)

h_order_items_grouped = h_order_items_products.groupBy("order_id").agg(
    
    avg("product_description_lenght").alias("average_product_desc_length"),
    avg("product_name_lenght").alias("average_product_name_length"),  
    round(F.first("total_items_in_order"),2).alias("total_items_in_order"),       
    round(F.first("total_price_of_order"),2).alias("total_price_of_order"),
    round(F.first("total_shipping_of_order"),2).alias("total_shipping_of_order"),
    sum('product_weight_g').alias('sum_of_all_products_weight'),
    F.first("product_category_name").alias("category"),
    sum('product_photos_qty').alias('total_photos_in_order'),
    sum('product_volume').alias('total_product_volume')        
)

# COMMAND ----------

h_order_items_products = h_order_items_grouped

# COMMAND ----------

bucketBorders = [3.0, 350.0, 600.0, 985.0, 4000.0]
h_order_items_products = Bucketizer(inputCol="average_product_desc_length", outputCol="Buckets_descr",splits=bucketBorders).transform(h_order_items_products)
bucketBorders_name = [3.0, 50.0, 100.0]
h_order_items_products = Bucketizer(inputCol="average_product_name_length", outputCol="Buckets_name",splits=bucketBorders_name).transform(h_order_items_products)
bucketBorders_weight = [0.0, 300.0, 800.0, 2000.0, 150000.0]
h_order_items_products = Bucketizer(inputCol="sum_of_all_products_weight", outputCol="Buckets_weight",splits=bucketBorders_weight).transform(h_order_items_products)
bucketBorders_volume = [0.0, 3000.0, 8000.0, 20000.0, 900000.0]
h_order_items_products = Bucketizer(inputCol="total_product_volume", outputCol="Buckets_volume",splits=bucketBorders_volume).transform(h_order_items_products)

# COMMAND ----------

columns_to_drop = [
    "average_product_desc_length", 
    "average_product_name_length", 
    "sum_of_all_products_weight", 
    "total_product_volume"
]

h_order_items_products = h_order_items_products.drop("average_product_desc_length", 
    "average_product_name_length", 
    "sum_of_all_products_weight", 
    "total_product_volume")

# COMMAND ----------

h_orders = h_orders.where(col("order_status")== "delivered").select("*")

# COMMAND ----------

h_orders = h_orders.drop("order_status")

# COMMAND ----------

h_orders = h_orders.dropna()

# COMMAND ----------

h_orders = h_orders.withColumn(
    "delivery_time",
    when(
        col("order_purchase_timestamp").isNotNull() & col("order_delivered_customer_date").isNotNull(),
        round(((unix_timestamp(col("order_delivered_customer_date")) - unix_timestamp(col("order_purchase_timestamp"))) / 86400).cast("double"), 2)
    ).otherwise(-1)
)

# COMMAND ----------

h_orders = h_orders.dropna()

# COMMAND ----------

# Convert date columns to timestamp format
h_orders = h_orders.withColumn("order_delivered_customer_date", to_timestamp(col("order_delivered_customer_date"))) \
               .withColumn("order_estimated_delivery_date", to_timestamp(col("order_estimated_delivery_date")))

# Add calculated columns
h_orders = h_orders \
    .withColumn(
        "delivered_on_time",
        when(col("order_delivered_customer_date") <= col("order_estimated_delivery_date"), True).otherwise(False)
    ) \
    .withColumn(
        "estimated_vs_actual_delay_days",
        round((unix_timestamp(col("order_delivered_customer_date")) - unix_timestamp(col("order_estimated_delivery_date"))) / 86400, 2)
    ) \
    .withColumn(
        "delivered_early",
        when(col("order_delivered_customer_date") < col("order_estimated_delivery_date"), True).otherwise(False)
    ) \
    .withColumn(
        "approval_time",
        when(
            col("order_purchase_timestamp").isNotNull() & col("order_approved_at").isNotNull(),
            round((unix_timestamp(col("order_approved_at")) - unix_timestamp(col("order_purchase_timestamp"))) / 86400, 2)
        ).otherwise(-1)
    ) \
    .withColumn(
        "delay_status",
        when(unix_timestamp(col("order_delivered_customer_date")) > unix_timestamp(col("order_estimated_delivery_date")), True).otherwise(False)
    )

# COMMAND ----------

h_orders= h_orders.dropna()

# COMMAND ----------

from pyspark.sql.functions import col, when, max, sum, round

# Step 1: Create "voucher_used" column to identify if a voucher was used in a payment
h_order_payments = h_order_payments.withColumn(
    "voucher_used",
    when(col("payment_type") == "voucher", 1).otherwise(0)
)

# Step 2: Aggregate voucher-related metrics
voucher_used = h_order_payments.groupBy("order_id").agg(
    max("voucher_used").alias("if_voucher_used")
)

voucher_count_df = h_order_payments.groupBy("order_id").agg(
    sum("voucher_used").alias("voucher_count")
)

sum_voucher_payments = h_order_payments.where(col("payment_type") == "voucher").groupBy("order_id").agg(
    round(sum("payment_value"), 2).alias("total_voucher_payment")
)

# Step 3: Aggregate non-voucher-related metrics
sum_non_voucher_payments = h_order_payments.where(col("payment_type") != "voucher").groupBy("order_id").agg(
    round(sum("payment_value"), 2).alias("total_non_voucher_payment")
)

# Step 4: Calculate maximum sequential and installment payments
max_payment_sequential = h_order_payments.groupBy("order_id").agg(
    max("payment_sequential").alias("max_payment_sequential")
)

max_payment_installments = h_order_payments.groupBy("order_id").agg(
    sum("payment_installments").alias("max_payment_installments")
)

# Step 5: Identify primary payment types
primary_payment_types = h_order_payments.where(col("payment_sequential") == 1).select(
    col("order_id"), col("payment_type").alias("primary_payment_type")
)

# Step 6: Combine all metrics into a clean DataFrame
h_order_payments_clean = voucher_used.join(voucher_count_df, on="order_id", how="inner") \
    .join(max_payment_sequential, on="order_id", how="inner") \
    .join(sum_voucher_payments, on="order_id", how="left") \
    .join(sum_non_voucher_payments, on="order_id", how="left") \
    .join(primary_payment_types, on="order_id", how="left") \
    .join(max_payment_installments, on="order_id", how="inner") \
    .filter(col("primary_payment_type").isNotNull())

# COMMAND ----------

h_order_payments_clean_nonulls = h_order_payments_clean.fillna({"total_voucher_payment": 0,"total_non_voucher_payment": 0})

# COMMAND ----------

h_basetable = (
    h_orders
    .join(h_order_payments_clean_nonulls, on="order_id", how="inner")
    .join(h_order_items_products, on="order_id", how="inner")
    
)

# COMMAND ----------

h_basetable.groupBy('primary_payment_type').count().show()

# COMMAND ----------

h_basetable.columns

# COMMAND ----------

# MAGIC %md
# MAGIC We fit the pipeline on the original table so that the structure is the same and we transform our holdout_basetable (h_basetable) on the basis of that. Our holdout data had fewer product categories compared to our training data so we had to fit it based on our training data.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder

# Step 2: StringIndexer for 'primary_payment_type'
payment_type_indexer = StringIndexer(inputCol="primary_payment_type", outputCol="primary_payment_type_index")

# Step 3: OneHotEncoder for the indexed 'primary_payment_type'
payment_type_encoder = OneHotEncoder(inputCol="primary_payment_type_index", outputCol="primary_payment_type_vector")

# Step 4: StringIndexer for the exploded 'category' column
categories_indexer = StringIndexer(inputCol="category", outputCol="categories_index")

# Step 5: OneHotEncoder for the indexed 'category' column
categories_encoder = OneHotEncoder(inputCol="categories_index", outputCol="categories_vector")

# Step 6: Combine all transformations in a Pipeline
pipeline = Pipeline(stages=[payment_type_indexer, payment_type_encoder, categories_indexer, categories_encoder])

# Step 7: Fit the pipeline to the training data (basetable2)
pipeline_model = pipeline.fit(basetable2)

# Step 8: Transform the holdout data (h_basetable) using the fitted pipeline model
vectorized_df_holdout = pipeline_model.transform(h_basetable)

# Drop the original columns that were indexed or encoded
vectorized_df_holdout = vectorized_df_holdout.drop('primary_payment_type', 'primary_payment_type_index', 'category', 'categories_index')

# Show the result for the holdout data
vectorized_df_holdout.show()

# COMMAND ----------

dup_hold = vectorized_df_holdout

# COMMAND ----------

vectorized_df_holdout = vectorized_df_holdout.drop(
    
    "order_purchase_timestamp", 
    "order_approved_at", 
    "order_delivered_carrier_date", 
    "order_delivered_customer_date", 
    "order_estimated_delivery_date", 
    "customer_id", 
    "review_id", 
    'categories_index',
    'categories',
    'primary_payment_type_index',
    'order_count',
    'review_score',
    'multiclass_label'
    )



# COMMAND ----------

# MAGIC %md
# MAGIC Now we run it thru Rformula and transform based on the fit from the training data.

# COMMAND ----------

from pyspark.ml.feature import RFormula

# Step 1: Fit the RFormula on the training data (basetable2)
r_form_binary = RFormula(formula="binary_label ~ . -order_id")  # Formula for binary classification
r_form_binary_model = r_form_binary.fit(basetable_binary)  # Fit on training data

r_form_multiclass = RFormula(formula="multiclass_label ~ . -order_id")  # Formula for multiclass classification
r_form_multiclass_model = r_form_multiclass.fit(basetable_multiclass)  # Fit on training data

# Step 2: Transform the holdout data (without labels)
r_form_df_binary_h_holdout = r_form_binary_model.transform(vectorized_df_holdout).select("order_id", "features")
r_form_df_multiclass_h_holdout = r_form_multiclass_model.transform(vectorized_df_holdout).select("order_id", "features")

# Show results for holdout data
r_form_df_binary_h_holdout.show()
r_form_df_multiclass_h_holdout.show()



# COMMAND ----------

# MAGIC %md
# MAGIC Predictions on holdout data.

# COMMAND ----------

predictions = gbt_model.transform(r_form_df_binary_h_holdout)


predictions.select('order_id',"features", "prediction", "probability").show()


predictions.select("prediction").show()



# COMMAND ----------

# MAGIC %md
# MAGIC Exporting the results.

# COMMAND ----------

final_holdover = predictions.select('order_id',"prediction")
df = pd.DataFrame(final_holdover.collect(), columns=final_holdover.columns)

# COMMAND ----------

df.head()

# COMMAND ----------

df.to_csv('C:/Users/omgit/Desktop/Big Data Proj/holdout_predictions.csv', index=False)
