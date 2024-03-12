# 003_POS_Tagging_Semantic_Similarity.py:
- Tokenizes the text and performs POS tagging using NLTK
- Replaces words with their synonyms based on POS tags
- Calculates semantic similarity using word vectors (wiki-news-300d-1M.vec)
- Selects top synonyms based on semantic similarity for replacement
- Skips words with length <= 3, containing hyphens, or proper nouns