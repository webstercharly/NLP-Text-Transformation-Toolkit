import argparse
import os
import random
import re
from functools import lru_cache
import nltk
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sentence_transformers import SentenceTransformer
from language_tool_python import LanguageTool
import markdown
from bs4 import BeautifulSoup

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

word_vectors = KeyedVectors.load_word2vec_format('./wiki-news-300d-1M.vec', binary=False)

stop_words = set(stopwords.words('english'))

bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
language_tool = LanguageTool('en-US')

@lru_cache(maxsize=1000)
def get_synonyms(word, pos):
    synonyms = []
    for syn in wordnet.synsets(word):
        if syn.pos() == pos:
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
    return synonyms

def calculate_similarity(sentence1, sentence2):
    embedding1 = sentence_transformer.encode([sentence1])[0]
    embedding2 = sentence_transformer.encode([sentence2])[0]
    return nltk.cluster.util.cosine_distance(embedding1, embedding2)

def replacement_prob(word, freq, importance):
    if word.lower() in stop_words:
        return 0.1

    prob = importance * (1 - freq / len(word_freq))
    return min(prob, 0.9)

def replace_with_synonyms(text_chunk):
    words = nltk.word_tokenize(text_chunk)
    pos_tags = nltk.pos_tag(words)
    modified_words = []

    for word, pos in pos_tags:
        if len(word) <= 3 or '-' in word or word.lower() in stop_words:
            modified_words.append(word)
        else:
            freq = word_freq[word.lower()]
            importance = freq / len(word_freq)
            if random.random() < replacement_prob(word, freq, importance):
                synonyms = get_synonyms(word, pos[0].lower())
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
    corrected_text = language_tool.correct(text)
    corrected_text = re.sub(r'\s+([.,!?])', r'\1', corrected_text)
    corrected_text = re.sub(r'\s+', ' ', corrected_text)
    corrected_text = '. '.join(sentence.capitalize() for sentence in corrected_text.split('. '))
    return corrected_text

def load_markdown_files(directory):
    text_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                html = markdown.markdown(content)
                text = ' '.join(BeautifulSoup(html, 'html.parser').find_all(string=True))
                text_data.append(text)
    return text_data

def fine_tune_bart(model, tokenizer, train_data, output_dir, num_train_epochs=3):
    train_encodings = tokenizer(train_data, truncation=True, padding=True)
    train_dataset = train_encodings['input_ids']

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)

def main():
    parser = argparse.ArgumentParser(description='Text Rewriter Tool')
    parser.add_argument('input_file', help='Path to the input text file')
    parser.add_argument('output_file', help='Path to the output text file')
    parser.add_argument('markdown_dir', help='Path to the directory containing markdown files for fine-tuning')

    args = parser.parse_args()

    try:

        with open(args.input_file, 'r', encoding='utf-8') as file:
            text = file.read()

        print("Original text:")
        print(text)

        model_output_dir = './fine_tuned_bart'
        if not os.path.exists(model_output_dir):
            markdown_data = load_markdown_files(args.markdown_dir)
            fine_tune_bart(bart_model, bart_tokenizer, markdown_data, output_dir=model_output_dir)
        else:
            bart_model = BartForConditionalGeneration.from_pretrained(model_output_dir)
            bart_tokenizer = BartTokenizer.from_pretrained(model_output_dir)

        global word_freq
        word_freq = nltk.FreqDist(nltk.word_tokenize(text.lower()))

        modified_sentences = []
        for sentence in sent_tokenize(text):
            modified_sentence = replace_with_synonyms(sentence)

            input_ids = bart_tokenizer.encode(modified_sentence, return_tensors='pt')
            output_ids = bart_model.generate(input_ids, max_length=len(input_ids[0])+50, num_beams=4, early_stopping=True)
            rephrased_sentence = bart_tokenizer.decode(output_ids[0], skip_special_tokens=True)

            similarity_threshold = 0.7
            if calculate_similarity(sentence, rephrased_sentence) < similarity_threshold:
                rephrased_sentence = modified_sentence

            modified_sentences.append(rephrased_sentence)

        modified_text = ' '.join(modified_sentences)
        modified_text = post_process(modified_text)

        print("\nModified text:")
        print(modified_text)

        with open(args.output_file, 'w', encoding='utf-8') as file:
            file.write(modified_text)

        print(f"\nModified text has been written to {args.output_file}")
    except FileNotFoundError:
        print(f"File not found: {args.input_file}")
    except IOError:
        print(f"An error occurred while reading or writing the file.")

if __name__ == '__main__':
    main()
