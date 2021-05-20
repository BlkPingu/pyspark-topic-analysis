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
import pickle
import os

if __name__ == '__main__':
    if not os.path.exists('documents.p'):
        # Import and preprocess documents
        raw_documents = []
        for file in glob.glob("texts/*.txt"):
            with open(file, 'r', encoding='utf-8') as f:
                raw_documents.append(f.read().replace('\n', ' '))

        nlp = spacy.load('en_core_web_sm', exclude=['ner'])
        nlp.max_length = 9999999

        documents = []
        for doc in nlp.pipe(raw_documents, n_process=-1, batch_size=1):
            documents.append([token.lemma_.lower() for token in doc
                              if token.pos_ == 'NOUN'
                              and token.is_alpha
                              and not token.is_stop
                              and len(token.lemma_) > 3])

        with open('documents.p', 'wb') as f:
            pickle.dump(documents, f)
    else:
        with open('documents.p', 'rb') as f:
            documents = pickle.load(f)

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
    idf = IDF(inputCol='tf_features', outputCol='tf_idf_features', minDocFreq=2)
    idf_model = idf.fit(tf_result)
    tfidf_result = idf_model.transform(tf_result)

    # Calculate topics
    lda = LDA(k=20, maxIter=10, featuresCol='tf_idf_features')
    lda_model = lda.fit(tfidf_result)

    vocab = tf_model.vocabulary
    udf_to_words = functions.udf(lambda token_list: [vocab[token_id] for token_id in token_list],
                                 types.ArrayType(types.StringType()))

    topics = lda_model.describeTopics(10).withColumn('topicWords', udf_to_words(functions.col('termIndices')))
    topics.select('topic', 'topicWords').show(truncate=200)
