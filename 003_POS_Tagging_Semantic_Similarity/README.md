# 003_POS_Tagging_Semantic_Similarity

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.0+-orange.svg)
![Gensim](https://img.shields.io/badge/gensim-4.0+-red.svg)

## Features
- Tokenizes the text and performs POS tagging using NLTK
- Replaces words with their synonyms based on POS tags
- Calculates semantic similarity using word vectors (wiki-news-300d-1M.vec)
- Selects top synonyms based on semantic similarity for replacement
- Skips words with length <= 3, containing hyphens, or proper nouns
