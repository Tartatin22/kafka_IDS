from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from joblib import dump

spark: SparkSession = SparkSession.builder.getOrCreate()

df = spark.read.csv('./data/KDDTrain.csv', header=False, inferSchema=True)

df = df.toDF(*["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
                   "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                   "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                   "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                   "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                   "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                   "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                   "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                   "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                   "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "labels", "label"])

# Sélectionner les colonnes nécessaires pour l'entraînement du modèle
selected_columns = ["src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "protocol_type_encoded",
                    "service_encoded", "flag_encoded"]

stages = []

# Transformer les colonnes catégorielles en numérique
indexer = StringIndexer(
    inputCols=["protocol_type", "service", "flag"], 
    outputCols=["protocol_type_indexed", "service_indexed", "flag_indexed"]
)

# Transformer les colonnes numériques en vecteurs binaires
encoder = OneHotEncoder(
    inputCols=["protocol_type_indexed", "service_indexed", "flag_indexed"], 
    outputCols=["protocol_type_encoded", "service_encoded", "flag_encoded"]
)

# Transformer les colonnes en vecteurs
assembler = VectorAssembler(inputCols=selected_columns, outputCol="features")
stages += [indexer, encoder, assembler]

# Transformer le labels en binaire (1 pour les attaques, 0 pour les non-attaques "normal" )
df_binary = df.withColumn("label", (df.labels != "normal").cast("integer"))

# Fit pipeline
pipeline = Pipeline(stages=stages)
df_binary = pipeline.fit(df_binary).transform(df_binary)

# Split dataset
(training_data_binary, test_data_binary) = df_binary.randomSplit([0.7, 0.3])

# Model Implementation
rf_binary = RandomForestClassifier(featuresCol="features", labelCol="label")
model_binary = rf_binary.fit(training_data_binary)

# Save model
model_binary.write().overwrite().save("./model/model_binary")

# Predictions
predictions_binary = model_binary.transform(test_data_binary)

# Calculer les métriques de classification
binary_evaluator = BinaryClassificationEvaluator(labelCol="label")
binary_accuracy = binary_evaluator.evaluate(predictions_binary, {binary_evaluator.metricName: "areaUnderROC"})

print("Binary Accuracy: {:.2%}".format(binary_accuracy))
