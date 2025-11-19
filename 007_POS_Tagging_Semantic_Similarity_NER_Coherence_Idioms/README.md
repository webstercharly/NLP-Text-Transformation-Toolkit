# 007_POS_Tagging_Semantic_Similarity_NER_Coherence_Idioms

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.0+-orange.svg)
![Spacy](https://img.shields.io/badge/spaCy-3.0+-blue.svg)

## Features
- Tokenizes the text and performs POS tagging using NLTK
- Replaces words with synonyms or related words based on semantic similarity
- Varies replacement probability based on word frequency and importance
- Identifies proper nouns and named entities using POS tags and Named Entity Recognition (NER)
- Detects and preserves idioms and phrasal verbs using an API
- Checks sentence-level coherence and grammatical errors using LanguageTool
- Skips words with length <= 3, containing hyphens, or identified as named entities
- Provides verbose output to track word replacements and skipped replacements
