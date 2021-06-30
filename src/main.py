import visualization
import data_processing
from os import path
import pickle
import text_processing

if __name__ == '__main__':
    # Pre-process or load documents
    if path.exists('documents.p'):
        with open('documents.p', 'rb') as f:
            documents = pickle.load(f)
    else:
        documents = text_processing.get_documents()
        with open('documents.p', 'wb') as f:
            pickle.dump(documents, f)

    for file_name, document in documents.items():
        # Calculate tf, tfidf and lda
        document_df, tf_model, tfidf_result, lda_model = data_processing.process_document(document)

        # Write pyLDAvis visualization to file
        visualization.generate_visualization(file_name, document_df, tf_model, tfidf_result, lda_model)

        # Basic output to console
        visualization.basic_output(tf_model, lda_model)
