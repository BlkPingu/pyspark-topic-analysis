# Inspired from https://stackoverflow.com/a/56545245
import numpy
from pyspark.sql import functions
from pyspark.sql import types
import pyLDAvis


def generate_visualization(df_filtered, count_vectorizer, tfidf_result, lda_model):
    tfidf_result_transformed = lda_model.transform(tfidf_result)

    data = __convert_data(df_filtered, count_vectorizer, tfidf_result_transformed, lda_model)
    with open('vis.html', 'w') as ff:
        pyLDAvis.save_html(pyLDAvis.prepare(**data), ff)


def __convert_data(df_filtered, count_vectorizer, transformed, lda_model):
    grouped_data = df_filtered.select((functions.explode(df_filtered.words)).alias("words")).groupby("words").count()
    word_counts = {r['words']: r['count'] for r in grouped_data.collect()}
    word_counts = [word_counts[w] for w in count_vectorizer.vocabulary]

    data = {'topic_term_dists': numpy.array(lda_model.topicsMatrix().toArray()).T,
            'doc_topic_dists': numpy.array(
                [x.toArray() for x in transformed.select(["topicDistribution"]).toPandas()['topicDistribution']]),
            'doc_lengths': [r[0] for r in df_filtered.select(functions.size(df_filtered.words)).collect()],
            'vocab': count_vectorizer.vocabulary,
            'term_frequency': word_counts}

    return __filter(data)


def __filter(data):
    doc_topic_dists = []
    doc_lengths = []

    for dist, length in zip(data['doc_topic_dists'], data['doc_lengths']):
        if numpy.sum(dist) == 1 and not numpy.isnan(dist).any():
            doc_topic_dists.append(dist)
            doc_lengths.append(length)

    data['doc_topic_dists'] = doc_topic_dists
    data['doc_lengths'] = doc_lengths

    return data


def basic_output(tf_model, lda_model):
    vocab = tf_model.vocabulary
    udf_to_words = functions.udf(lambda token_list: [vocab[token_id] for token_id in token_list],
                                 types.ArrayType(types.StringType()))
    topics = lda_model.describeTopics(10).withColumn('topicWords', udf_to_words(functions.col('termIndices')))
    topics.select('topic', 'topicWords').show(truncate=200)
