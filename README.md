# Twitter Sentiment Analysis

A classical NLP pipeline for classifying tweets as regular/positive or negative (racist/sexist) sentiment. The project covers the full workflow from raw text to a trained classifier: cleaning noisy tweet text, exploring hashtag and word patterns by sentiment class, extracting Bag-of-Words and TF-IDF features, and training a Logistic Regression baseline.

## Overview

Tweets are short, informal, and full of noise (`@mentions`, punctuation, abbreviations) that isn't useful signal for sentiment classification. This project builds a preprocessing pipeline to strip that noise out, visualizes what's left to sanity-check the cleaning and to see which hashtags are associated with each sentiment class, then converts the cleaned text into numeric features a model can consume, and trains a Logistic Regression classifier to predict sentiment.

## Method

1. **Text preprocessing** — For each tweet: strip `@user` mentions via regex, remove all non-alphabetic characters (keeping `#` so hashtags survive), drop short/low-signal words (length &le; 3), tokenize, and apply Porter stemming to normalize word variants (e.g. "loving"/"loved" -> "love").
2. **Exploratory analysis** — Word clouds of the most frequent terms overall and per sentiment class, plus bar charts comparing the top hashtags used in regular vs. negative-sentiment tweets.
3. **Feature extraction** — Two numeric representations of the cleaned text, each capped at 1000 features with English stop words removed:
   - **Bag-of-Words** (`CountVectorizer`) — raw term counts.
   - **TF-IDF** (`TfidfVectorizer`) — term counts re-weighted by how distinctive a word is across the corpus.
4. **Model training & evaluation** — A `LogisticRegression` classifier is trained on the Bag-of-Words features (70/30 train/validation split), with predictions thresholded at 0.3 on the positive-class probability rather than the default 0.5, to favor recall on the minority negative-sentiment class.

## Results

The Logistic Regression baseline achieves an **F1-score of approximately 0.53** on the held-out validation split.

This is a modest score, typical of a classical Bag-of-Words + Logistic Regression baseline on this dataset. It establishes a reasonable starting point rather than a final answer. Natural next steps to improve on it:

- Word embeddings (Word2Vec, GloVe, or fastText) in place of sparse BoW/TF-IDF features, which capture semantic similarity between words.
- Transformer-based models (e.g. BERT and its variants), which are the current standard for tweet-level sentiment classification and would be expected to meaningfully outperform this baseline.
- Handling class imbalance more explicitly (the negative class is a small minority of the dataset) via resampling or class-weighted loss.

## Dataset

This project uses Analytics Vidhya's **"Twitter Sentiment Analysis"** practice problem dataset (`train.csv` and `test.csv`). The dataset is not included in this repository — download it from Analytics Vidhya's practice problems section and place `train.csv` and `test.csv` in the repository root before running the notebook.

## Tech Stack

- Python
- pandas, numpy
- nltk (tokenization, Porter stemming, hashtag/word frequency analysis)
- scikit-learn (`CountVectorizer`, `TfidfVectorizer`, `LogisticRegression`, `train_test_split`, `f1_score`)
- matplotlib, seaborn, wordcloud (visualization)

## How to Run

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud jupyter
   ```
3. Download `train.csv` and `test.csv` from Analytics Vidhya's Twitter Sentiment Analysis practice problem and place them in the repository root (same directory as the notebook).
4. Launch Jupyter and run the notebook top to bottom:
   ```bash
   jupyter notebook "Twitter Sentiment Analysis.ipynb"
   ```
