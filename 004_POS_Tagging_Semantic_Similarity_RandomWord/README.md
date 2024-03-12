# 004_POS_Tagging_Semantic_Similarity_RandomWord.py:
- Tokenizes the text and performs POS tagging using NLTK
- Replaces words with their synonyms based on POS tags and semantic similarity
- Calculates word frequencies and varies replacement probability based on frequency
- Varies word order by randomly swapping two words in a sentence
- Skips words with length <= 3, containing hyphens, or proper nouns