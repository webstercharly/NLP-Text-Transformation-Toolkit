# 008_GPT2_BART_Coherence
- Tokenizes input text into sentences and words, employing NLTK for basic linguistic processing.
- Identifies synonyms for word replacement based on context, using WordNet for synonym retrieval.
- GPT-2 to generate multiple rephrased versions of each sentence, enhancing variety and fluency.
- Uses BERT's next-sentence prediction to evaluate and select the best rephrased sentence that maintains coherence with the preceding text.
- Applies a post-processing step to clean up the text formatting, ensuring proper punctuation and spacing.