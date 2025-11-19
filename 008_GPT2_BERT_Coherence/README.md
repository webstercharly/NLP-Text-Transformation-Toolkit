# 008_T5_BERT_Coherence

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Transformers](https://img.shields.io/badge/transformers-4.0+-orange.svg)

## Features
- Tokenizes input text into sentences and words, employing NLTK for basic linguistic processing
- Identifies synonyms for word replacement based on context, using WordNet for synonym retrieval
- T5 model to generate paraphrased versions of each sentence with proper text-to-text transformation
- Uses BERT's next-sentence prediction to evaluate and select the best rephrased sentence that maintains coherence with the preceding text
- Sentence transformers for accurate semantic similarity measurement between sentences
- Applies a post-processing step to clean up the text formatting, ensuring proper punctuation and spacing
