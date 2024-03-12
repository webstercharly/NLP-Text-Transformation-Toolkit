import argparse
import random
import re
from functools import lru_cache
import nltk
from nltk.corpus import wordnet
from gensim.models import KeyedVectors

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#word_vectors = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
word_vectors = KeyedVectors.load_word2vec_format('./wiki-news-300d-1M.vec', binary=False)

@lru_cache(maxsize=1000)
def get_synonyms(word, pos):
    synonyms = []
    for syn in wordnet.synsets(word):
        if syn.pos() == pos:
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
    return synonyms

def calculate_similarity(word1, word2):
    if word1 in word_vectors and word2 in word_vectors:
        return word_vectors.similarity(word1, word2)
    else:
        return 0.0

def replace_with_synonyms(text_chunk):
    words = nltk.word_tokenize(text_chunk)
    pos_tags = nltk.pos_tag(words)
    modified_words = []

    for i, (word, pos) in enumerate(pos_tags):
        if len(word) <= 3 or '-' in word or pos.startswith('NNP'):
            modified_words.append(word)
            continue
        if i % random.randint(2, 4) == 0:
            pos_category = pos[0].lower()
            synonyms = get_synonyms(word, pos_category)
            if synonyms:
                top_synonyms = sorted(synonyms, key=lambda x: calculate_similarity(word, x), reverse=True)[:5]
                synonym = random.choice(top_synonyms)
                modified_words.append(synonym)
            else:
                modified_words.append(word)
        else:
            modified_words.append(word)

    modified_text = ' '.join(modified_words)
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

        modified_text = replace_with_synonyms(text)
        modified_text = post_process(modified_text)

        with open(args.output_file, 'w') as file:
            file.write(modified_text)

        print(f"Modified text has been written to {args.output_file}")
    except FileNotFoundError:
        print(f"File not found: {args.input_file}")
    except IOError:
        print(f"An error occurred while reading or writing the file.")

if __name__ == '__main__':
    main()
