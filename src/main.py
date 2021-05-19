import os
import glob
import spacy
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.sql import types
from pyspark.sql import Row
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from pyspark.ml.clustering import LDA

nlp = spacy.load('en_core_web_sm')
nlp.max_length = 9999999
nlp.add_pipe('merge_noun_chunks')

os.environ["HADOOP_HOME"] = r'C:\Users\Roman\Desktop\sparkproto\spark'  # is that even needed?

# Import and preprocess documents
documents = []
for file in glob.glob("texts/*.txt"):
    with open(file, 'r', encoding='utf-8') as f:
        print(file)
        text = f.read().replace('\n', ' ')
        doc = nlp(text)
        documents.append([token.lemma_.lower() for token in doc if token.pos_ in ['NOUN', 'VERB']])

spark_context = SparkContext(appName="TF-IDF & LDA Prototype")
spark_session = SparkSession(spark_context)

# Create DataFrame
rdd1 = spark_context.parallelize(documents)
row_rdd = rdd1.map(lambda x: Row(x))
documents_df = spark_session.createDataFrame(row_rdd, ['words'])

# https://github.com/maobedkova/TopicModelling_PySpark_SparkNLP/blob/master/Topic_Modelling_with_PySpark_and_Spark_NLP.ipynb

# Calculate tf
tf = CountVectorizer(inputCol='words', outputCol='tf_features')
tf_model = tf.fit(documents_df)
tf_result = tf_model.transform(documents_df)

# Calculate tf-idf
idf = IDF(inputCol='tf_features', outputCol='tf_idf_features')
idf_model = idf.fit(tf_result)
tfidf_result = idf_model.transform(tf_result)

# Calculate topics
lda = LDA(k=6, maxIter=10, featuresCol='tf_idf_features')
lda_model = lda.fit(tfidf_result)

udf_to_words = functions.udf(lambda token_list: [tf_model.vocabulary[token_id] for token_id in token_list],
                             types.ArrayType(types.StringType()))

topics = lda_model.describeTopics(10).withColumn('topicWords', udf_to_words(functions.col('termIndices')))
topics.select('topic', 'topicWords').show(truncate=90)
