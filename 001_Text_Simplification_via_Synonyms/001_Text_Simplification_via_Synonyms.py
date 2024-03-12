import argparse
import random
import requests
import multiprocessing
import re
from functools import lru_cache

with open('dale_chall_word_list.txt', 'r') as file:
    dale_chall_words = set(word.strip().lower() for word in file)

@lru_cache(maxsize=1000)
def get_simpler_word(word):
    if word.lower() in dale_chall_words:
        print(f"Word is already simple: {word}")
        return word

    url = f"https://api.datamuse.com/words?rel_syn={word}&md=f"
    response = requests.get(url)

    if response.status_code == 200:
        synonyms_data = response.json()
        if synonyms_data:
            simpler_synonyms = [syn['word'] for syn in synonyms_data if 'score' in syn and int(syn['score']) >= 1000]
            if simpler_synonyms:
                return random.choice(simpler_synonyms)
    return word

def replace_with_simpler_words(text_chunk):
    words = re.findall(r'\b(?![\w-]+-)(?![A-Z])[\w-]+\b', text_chunk)
    modified_text = text_chunk

    for i, word in enumerate(words):
        if len(word) <= 3 or '-' in word:
            continue
        if i % random.randint(2, 4) == 0:
            simpler_word = get_simpler_word(word)
            if simpler_word != word:
                print(f"Replacing complex word: {word} ({i+1}/{len(words)})")
            modified_text = re.sub(r'\b{}\b'.format(re.escape(word)), simpler_word, modified_text)

    return modified_text

def process_chunk(chunk):
    return replace_with_simpler_words(chunk)

def post_process(text):
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def main():
    parser = argparse.ArgumentParser(description='Text Simplifier Tool')
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

        print(f"Simplified text has been written to {args.output_file}")
    except FileNotFoundError:
        print(f"File not found: {args.input_file}")
    except IOError:
        print(f"An error occurred while reading or writing the file.")

if __name__ == '__main__':
    main()
