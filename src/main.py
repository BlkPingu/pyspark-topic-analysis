import visualization
import data_processing
from os import path
import pickle
import text_processing

if __name__ == '__main__':
    # Import pre-processed documents
    if path.exists('documents.p'):
        with open('documents.p', 'rb') as f:
            documents = pickle.load(f)
    else:
        documents = text_processing.get_documents()
        with open('documents.p', 'wb') as f:
            pickle.dump(documents, f)

    # Process documents
    documents_df, tf_model, tfidf_result, lda_model = data_processing.process_documents(documents)

    # Write pyLDAvis visualization to file
    visualization.generate_visualization(documents_df, tf_model, tfidf_result, lda_model)

    # Basic output to console
    visualization.basic_output(tf_model, lda_model)
