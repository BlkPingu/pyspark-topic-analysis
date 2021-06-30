# Inspired from https://github.com/maobedkova/TopicModelling_PySpark_SparkNLP/blob/master/Topic_Modelling_with_PySpark_and_Spark_NLP.ipynb
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA
import os

spark_context = SparkContext()
spark_session = SparkSession(spark_context)

os.environ['PYSPARK_PYTHON'] = '.venv/Scripts/python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = '.venv/Scripts/python.exe'


# Set when running in cluster
# SparkContext.setSystemProperty('spark.master', 'spark://192.168.2.122:7077')
# SparkContext.setSystemProperty('spark.driver.memory', '32g')
# SparkContext.setSystemProperty('spark.executor.instances', '4')
# SparkContext.setSystemProperty('spark.executor.cores', '4')
# SparkContext.setSystemProperty('spark.executor.memory', '8g')
# SparkContext.setSystemProperty('spark.default.parallelism', '4')

def process_document(document):
    document_df = __create_dataframe(document)
    tf_model, tf_result = __calculate_tf(document_df)
    tfidf_result = __calculate_tfidf(tf_result)
    return document_df, tf_model, tfidf_result, __calculate_lda(tfidf_result)


def __create_dataframe(document):
    rdd = spark_context.parallelize(document)
    row_rdd = rdd.map(lambda x: Row(x))
    return spark_session.createDataFrame(row_rdd, ['words'])


def __calculate_tf(document_df):
    tf = CountVectorizer(inputCol='words', outputCol='tf_features')
    tf_model = tf.fit(document_df)
    return tf_model, tf_model.transform(document_df)


def __calculate_tfidf(tf_result):
    idf = IDF(inputCol='tf_features', outputCol='tf_idf_features', minDocFreq=2)
    idf_model = idf.fit(tf_result)
    return idf_model.transform(tf_result)


def __calculate_lda(tfidf_result):
    lda = LDA(featuresCol='tf_idf_features')
    return lda.fit(tfidf_result)
