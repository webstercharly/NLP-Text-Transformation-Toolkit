import argparse
import random
import re
from functools import lru_cache
import nltk
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from language_tool_python import LanguageTool
import requests
import spacy

# Download NLTK data with quiet flag
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

word_vectors = KeyedVectors.load_word2vec_format('./wiki-news-300d-1M.vec', binary=False)

stop_words = set(stopwords.words('english'))

API_KEY = ''

lt = LanguageTool('en-US')
nlp = spacy.load("en_core_web_sm")

def is_named_entity(word):
    doc = nlp(word)
    return any(token.ent_type_ for token in doc)

def is_idiom_or_phrasal_verb(phrase):
    url = 'https://api.api-ninjas.com/v1/idioms'
    headers = {'X-Api-Key': API_KEY}
    params = {'phrase': phrase}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        results = response.json()
        return len(results) > 0
    else:
        return False

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

def is_proper_noun_or_named_entity(word, pos):
    if pos.startswith('NNP') or pos.startswith('NNPS'):
        return True

    named_entities = ne_chunk(pos_tag([word]))
    for chunk in named_entities:
        # Check if chunk is a named entity (has label attribute)
        if hasattr(chunk, 'label'):
            # NLTK returns labels like 'PERSON', 'ORGANIZATION', 'GPE', etc.
            return True

    return False

def replacement_prob(word, freq, importance, word_freq_len):
    if word.lower() in stop_words:
        return 0.1

    prob = importance * (1 - freq / word_freq_len)
    return min(prob, 0.9)

def replace_with_synonyms(text_chunk, word_freq):
    words = nltk.word_tokenize(text_chunk)
    pos_tags = nltk.pos_tag(words)
    modified_words = []

    i = 0
    while i < len(pos_tags):
        word, pos = pos_tags[i]
        if len(word) <= 3 or '-' in word or is_named_entity(word):
            modified_words.append(word)
            i += 1
        else:
            # Check for phrasal verbs (verb + particle pattern)
            phrase = [word]
            j = i + 1
            # Look for verb followed by particle (RP), preposition (IN), or adverb (RB)
            if pos.startswith('VB') and j < len(pos_tags):
                next_pos = pos_tags[j][1]
                if next_pos in ['RP', 'IN', 'RB']:
                    phrase.append(pos_tags[j][0])
                    j += 1

            phrase_text = ' '.join(phrase)
            if len(phrase) > 1 and is_idiom_or_phrasal_verb(phrase_text):
                modified_words.append(phrase_text)
                print(f"Preserved idiom/phrasal verb: {phrase_text}")
                i = j
            else:
                # Process each word individually
                for w in phrase:
                    freq = word_freq[w.lower()]
                    importance = freq / len(word_freq)
                    if random.random() < replacement_prob(w, freq, importance, len(word_freq)):
                        synonyms = get_synonyms(w, pos[0].lower())
                        related_words = get_related_words(w)
                        replacements = synonyms + related_words
                        if replacements:
                            top_replacements = sorted(replacements, key=lambda x: calculate_similarity(w, x), reverse=True)[:5]
                            replacement = random.choice(top_replacements)
                            modified_words.append(replacement)
                            print(f"Replaced '{w}' with '{replacement}'")
                        else:
                            modified_words.append(w)
                            print(f"No suitable replacement found for '{w}'")
                    else:
                        modified_words.append(w)
                        print(f"Skipped replacement for '{w}'")

                i = j

    modified_text = ' '.join(modified_words)

    matches = lt.check(modified_text)
    if len(matches) > 0:
        print("Reverted to original sentence due to grammatical errors")
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

        word_freq = nltk.FreqDist(nltk.word_tokenize(text.lower()))

        modified_text = replace_with_synonyms(text, word_freq)
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
