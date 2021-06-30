import glob
import spacy
from pathlib import Path

nlp = spacy.load('de_core_news_sm')


def get_documents():
    documents = {}
    for file_name, raw_document in __import_raw_documents().items():
        documents[file_name] = __process_raw_document(raw_document)
    return documents


def __import_raw_documents():
    raw_documents = {}
    for file in glob.glob("texts/German/*.txt"):
        with open(file, 'r', encoding='utf-8') as f:
            file_name = Path(file).stem
            raw_documents[file_name] = f.read().split('\n')
    return raw_documents


def __process_raw_document(raw_document):
    document = []
    for doc in nlp.pipe([line for line in raw_document if len(line) > 3], n_process=-1):
        document.append([token.lemma_.lower() for token in doc
                         if token.pos_ == 'NOUN'
                         and token.is_alpha
                         and not token.is_stop
                         and len(token.lemma_) > 1])
    return [part for part in document if part]
