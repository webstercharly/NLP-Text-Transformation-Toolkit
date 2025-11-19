import argparse
import random
import re
from functools import lru_cache
import nltk
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertForNextSentencePrediction, BertTokenizer
from sentence_transformers import SentenceTransformer
import torch

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

word_vectors = KeyedVectors.load_word2vec_format('./wiki-news-300d-1M.vec', binary=False)

stop_words = set(stopwords.words('english'))

# Use T5 for paraphrasing (text-to-text model)
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

bert_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Use sentence transformer for semantic similarity
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

@lru_cache(maxsize=1000)
def get_synonyms(word, pos):
    synonyms = []
    for syn in wordnet.synsets(word):
        if syn.pos() == pos:
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
    return synonyms

def calculate_word_similarity(word1, word2):
    """Calculate similarity between two words using word vectors"""
    if word1 in word_vectors and word2 in word_vectors:
        return word_vectors.similarity(word1, word2)
    else:
        return 0.0

def calculate_sentence_similarity(sentence1, sentence2):
    """Calculate similarity between two sentences using sentence transformers"""
    embeddings = sentence_model.encode([sentence1, sentence2])
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        torch.tensor(embeddings[0]).unsqueeze(0),
        torch.tensor(embeddings[1]).unsqueeze(0)
    )
    return cos_sim.item()

def replacement_prob(word, freq, importance, word_freq_len):
    if word.lower() in stop_words:
        return 0.1

    prob = importance * (1 - freq / word_freq_len)
    return min(prob, 0.9)

def replace_with_synonyms(text_chunk, word_freq):
    words = nltk.word_tokenize(text_chunk)
    pos_tags = nltk.pos_tag(words)
    modified_words = []

    for word, pos in pos_tags:
        if len(word) <= 3 or '-' in word or word.lower() in stop_words:
            modified_words.append(word)
        else:
            freq = word_freq[word.lower()]
            importance = freq / len(word_freq)
            if random.random() < replacement_prob(word, freq, importance, len(word_freq)):
                synonyms = get_synonyms(word, pos[0].lower())
                if synonyms:
                    top_synonyms = sorted(synonyms, key=lambda x: calculate_word_similarity(word, x), reverse=True)[:5]
                    synonym = random.choice(top_synonyms)
                    modified_words.append(synonym)
                else:
                    modified_words.append(word)
            else:
                modified_words.append(word)

    modified_text = ' '.join(modified_words)
    return modified_text

def check_coherence(sentence1, sentence2):
    inputs = bert_tokenizer(sentence1, sentence2, return_tensors='pt')
    outputs = bert_model(**inputs)
    probabilities = outputs.logits.softmax(dim=-1)
    return probabilities[0][0].item()

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

        modified_sentences = []
        prev_sentence = None
        for sentence in sent_tokenize(text):
            modified_sentence = replace_with_synonyms(sentence, word_freq)

            # Generate paraphrases using T5
            rephrased_sentences = []
            for _ in range(3):
                # T5 uses task prefix for different tasks
                input_text = f"paraphrase: {modified_sentence}"
                input_ids = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
                output_ids = t5_model.generate(
                    input_ids,
                    max_length=len(input_ids[0])+20,
                    num_return_sequences=1,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )
                rephrased_sentence = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
                rephrased_sentences.append(rephrased_sentence)

            # Select best paraphrase based on similarity and coherence
            if prev_sentence is not None:
                best_rephrased = max(rephrased_sentences, key=lambda x: calculate_sentence_similarity(sentence, x) + check_coherence(prev_sentence, x))
            else:
                best_rephrased = max(rephrased_sentences, key=lambda x: calculate_sentence_similarity(sentence, x))

            modified_sentences.append(best_rephrased)
            prev_sentence = best_rephrased

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
