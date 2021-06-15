# Inspired from https://github.com/maobedkova/TopicModelling_PySpark_SparkNLP/blob/master/Topic_Modelling_with_PySpark_and_Spark_NLP.ipynb
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import LDA


def process_documents(documents):
    documents_df = __create_dataframe(documents)
    tf_model, tf_result = __calculate_tf(documents_df)
    tfidf_result = __calculate_tfidf(tf_result)
    return documents_df, tf_model, tfidf_result, __calculate_lda(tfidf_result)


def __create_dataframe(documents):
    spark_context = SparkContext()
    spark_session = SparkSession(spark_context)
    rdd1 = spark_context.parallelize(documents)
    row_rdd = rdd1.map(lambda x: Row(x))
    return spark_session.createDataFrame(row_rdd, ['words'])


def __calculate_tf(documents_df):
    tf = CountVectorizer(inputCol='words', outputCol='tf_features')
    tf_model = tf.fit(documents_df)
    return tf_model, tf_model.transform(documents_df)


def __calculate_tfidf(tf_result):
    idf = IDF(inputCol='tf_features', outputCol='tf_idf_features', minDocFreq=2)
    idf_model = idf.fit(tf_result)
    return idf_model.transform(tf_result)


def __calculate_lda(tfidf_result):
    lda = LDA(featuresCol='tf_idf_features')
    return lda.fit(tfidf_result)
