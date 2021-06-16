import glob
import spacy

nlp = spacy.load('de_core_news_sm')


def get_documents():
    documents = []
    for raw_document in __import_raw_documents():
        __process_raw_document(raw_document)
    return [document for document in documents if document]


def __import_raw_documents():
    raw_documents = []
    for file in glob.glob("texts/German/*.txt"):
        with open(file, 'r', encoding='utf-8') as f:
            raw_documents.append(f.read().split('\n'))
    return raw_documents


def __process_raw_document(raw_document):
    document = []
    for doc in nlp.pipe([line for line in raw_document if len(line) > 3], n_process=-1):
        document = document + [token.lemma_.lower() for token in doc
                               if token.pos_ == 'NOUN'
                               and token.is_alpha
                               and not token.is_stop
                               and len(token.lemma_) > 1]
    return document
