import argparse
import random
import requests
import multiprocessing
import re
from functools import lru_cache
import nltk
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

@lru_cache(maxsize=1000)
def get_synonyms(word, pos):
    synonyms = []
    for syn in wordnet.synsets(word):
        if syn.pos() == pos:
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
    return synonyms

def replace_with_synonyms(text_chunk):
    words = nltk.word_tokenize(text_chunk)
    pos_tags = nltk.pos_tag(words)
    modified_words = []

    for i, (word, pos) in enumerate(pos_tags):
        if len(word) <= 3 or '-' in word or pos.startswith('NNP'):
            modified_words.append(word)
            continue
        if i % random.randint(2, 4) == 0:
            print(f"Replacing synonyms for: {word} ({i+1}/{len(pos_tags)})")
            pos_category = pos[0].lower()
            synonyms = get_synonyms(word, pos_category)
            if synonyms:
                modified_words.append(random.choice(synonyms))
            else:
                modified_words.append(word)
        else:
            modified_words.append(word)

    modified_text = ' '.join(modified_words)
    return modified_text

def process_chunk(chunk):
    return replace_with_synonyms(chunk)

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

        chunks = re.split(r'(?<=\n)', text)

        with multiprocessing.Pool() as pool:
            modified_chunks = pool.map(process_chunk, chunks)

        modified_text = ''.join(modified_chunks)
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
