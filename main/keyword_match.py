import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
import re

# nltk.download('punkt')
# nltk.download('wordnet')

# Initialize the stemmer.
stemmer = PorterStemmer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return set(stemmed_words)


def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())  # Add synonyms
    return synonyms


def keywords_match(sentence1, sentence2):
    keywords1 = preprocess_text(sentence1)
    keywords2 = preprocess_text(sentence2)
    all_keywords1 = set()
    for word in keywords1:
        all_keywords1.update(get_synonyms(word))
    all_keywords1.update(keywords1)  # Add the original keywords

    all_keywords2 = set()
    for word in keywords2:
        all_keywords2.update(get_synonyms(word))
    all_keywords2.update(keywords2)  # Add the original keywords

    # Calculate the intersection
    intersection = all_keywords1.intersection(all_keywords2)

    # Calculate the matching score
    if len(all_keywords1) == 0 and len(all_keywords2) == 0:
        return 0.0  # Avoid division by zero

    # Calculate the matching score
    score = len(intersection) / max(len(all_keywords1), len(all_keywords2))
    return score


def keywords_match_quik(sentence1, sentence2):
    # Split the sentence and convert to a set
    words1 = set(sentence1.split())
    words2 = set(sentence2.split())

    # Calculate the intersection
    common_keywords = words1.intersection(words2)

    # Calculate the matching score
    if not words1 or not words2:  # Avoid division by zero
        return 0

    # Adjust the score calculation as needed
    score = len(common_keywords) / min(len(words1), len(words2))

    return score


