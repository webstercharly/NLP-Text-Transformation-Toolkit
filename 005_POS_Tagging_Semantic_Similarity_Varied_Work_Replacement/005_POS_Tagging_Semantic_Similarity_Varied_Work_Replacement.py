import argparse
import random
import re
from functools import lru_cache
import nltk
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

import spacy
import requests
from language_tool_python import LanguageTool

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

word_vectors = KeyedVectors.load_word2vec_format('./wiki-news-300d-1M.vec', binary=False)

stop_words = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_sm")

with open('domain_vocab.txt', 'r') as file:
    domain_vocab = set(word.strip().lower() for word in file)

lt = LanguageTool('en-US')

@lru_cache(maxsize=1000)
def get_synonyms(word, pos):
    synonyms = []
    for syn in wordnet.synsets(word):
        if syn.pos() == pos:
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
    return synonyms

def get_related_words(word):
    url = f"https://api.datamuse.com/words?ml={word}"
    response = requests.get(url)
    related_words = [item['word'] for item in response.json()[:5]]
    return related_words

def calculate_similarity(word1, word2):
    if word1 in word_vectors and word2 in word_vectors:
        return word_vectors.similarity(word1, word2)
    else:
        return 0.0

def is_named_entity(word):
    doc = nlp(word)
    return any(token.ent_type_ for token in doc)

def replacement_prob(word, freq, importance):
    if word.lower() in stop_words or word.lower() in domain_vocab:
        return 0.1

    prob = importance * (1 - freq / len(word_freq))
    return min(prob, 0.9)

def replace_with_synonyms(text_chunk):
    words = nltk.word_tokenize(text_chunk)
    pos_tags = nltk.pos_tag(words)
    modified_words = []

    for word, pos in pos_tags:
        if len(word) <= 3 or '-' in word or is_named_entity(word):
            modified_words.append(word)
        else:
            freq = word_freq[word.lower()]
            importance = freq / len(word_freq)
            if random.random() < replacement_prob(word, freq, importance):
                synonyms = get_synonyms(word, pos[0].lower())
                related_words = get_related_words(word)
                replacements = synonyms + related_words
                if replacements:
                    top_replacements = sorted(replacements, key=lambda x: calculate_similarity(word, x), reverse=True)[:5]
                    replacement = random.choice(top_replacements)
                    modified_words.append(replacement)
                else:
                    modified_words.append(word)
            else:
                modified_words.append(word)

    modified_text = ' '.join(modified_words)

    matches = lt.check(modified_text)
    if len(matches) > 0:
        modified_text = text_chunk

    return modified_text

def post_process(text):
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def main():
    parser = argparse.ArgumentParser(description='Text Rewriter Tool')
    parser.add_argument('input_file', help='Path to the input text file')
    parser.add_argument('output_file', help='Path to the output text file')

    args = parser.parse_args()

    try:
        with open(args.input_file, 'r') as file:
            text = file.read()

        print("Original text:")
        print(text)

        global word_freq
        word_freq = nltk.FreqDist(nltk.word_tokenize(text.lower()))

        sentences = nltk.sent_tokenize(text)

        modified_sentences = []
        for sentence in sentences:
            modified_sentence = replace_with_synonyms(sentence)
            modified_sentences.append(modified_sentence)

        modified_text = ' '.join(modified_sentences)
        modified_text = post_process(modified_text)

        print("\nModified text:")
        print(modified_text)

        with open(args.output_file, 'w') as file:
            file.write(modified_text)

        print(f"\nModified text has been written to {args.output_file}")
    except FileNotFoundError:
        print(f"File not found: {args.input_file}")
    except IOError:
        print(f"An error occurred while reading or writing the file.")

if __name__ == '__main__':
    main()
