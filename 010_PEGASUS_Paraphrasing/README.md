# 010_PEGASUS_Paraphrasing

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Transformers](https://img.shields.io/badge/transformers-4.0+-orange.svg)

## Overview
Advanced text paraphrasing using PEGASUS model combined with synonym replacement, BERT coherence checking, and semantic similarity scoring. PEGASUS is specifically pre-trained for paraphrase generation tasks.

## Features
- **PEGASUS Model**: Pre-trained paraphrase generation model (tuner007/pegasus_paraphrase)
- **Multiple Paraphrases**: Generate multiple paraphrase candidates
- **Coherence Checking**: BERT-based next sentence prediction for inter-sentence coherence
- **Semantic Similarity**: Sentence transformers for maintaining meaning
- **Synonym Replacement**: WordNet-based word replacement with semantic similarity filtering
- **Beam Search**: Configurable beam search for diverse paraphrasing options

## Requirements
```bash
pip install transformers sentence-transformers torch nltk gensim
```

## Usage
```bash
python 010_PEGASUS_Paraphrasing.py input.txt output.txt --num_beams 10 --num_return_sequences 3
```

### Parameters
- `input_file`: Path to input text file
- `output_file`: Path to output text file
- `--num_beams`: Number of beams for beam search (default: 10)
- `--num_return_sequences`: Number of paraphrases to generate per sentence (default: 3)

## Model Details
- **Paraphrasing**: tuner007/pegasus_paraphrase (PEGASUS fine-tuned for paraphrasing)
- **Coherence**: bert-base-uncased (Next Sentence Prediction)
- **Similarity**: all-MiniLM-L6-v2 (Sentence Transformers)
- **Word Vectors**: FastText wiki-news-300d-1M

## How It Works
1. Tokenizes input text into sentences
2. Applies synonym replacement with semantic filtering
3. Generates multiple paraphrases using PEGASUS
4. Scores each paraphrase based on:
   - Semantic similarity to original
   - Coherence with previous sentence
5. Selects best paraphrase for each sentence
6. Post-processes output for proper formatting
