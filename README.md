# LSTM NLP Assignment – Tweet Classification

This repository contains a Jupyter Notebook implementing an end-to-end NLP pipeline for tweet classification using deep learning (LSTM-based models). The focus of the notebook is **robust text preprocessing**, **data augmentation**, and **sequence modeling** for short, noisy social media text.

The notebook was originally developed and executed in a Kaggle environment.

---

## Overview

The notebook covers the following stages:

1. Loading and exploring a tweet dataset  
2. Advanced text preprocessing tailored for tweets  
3. Optional synonym-based data augmentation  
4. Preparing cleaned text for LSTM input  
5. Training and evaluating an LSTM-based neural network  

The preprocessing pipeline is designed specifically for **informal text** (hashtags, emojis, contractions, negations, etc.), which is common in Twitter data.

---

## Dataset

The notebook uses the **TweetEval** dataset:

- `tweet_eval_train.csv`
- `tweet_eval_validation.csv`
- `tweet_eval_test.csv`

Each file contains:
- `text`: raw tweet text
- `label`: target class (task-dependent)

> **Note:**  
> File paths are hardcoded for Kaggle:
> ```
> /kaggle/input/tweetsmou/
> ```
> If you run this locally, update paths accordingly.

---

## Text Preprocessing Pipeline

The preprocessing function `preprocess_for_lstm()` performs the following steps:

### Text Normalization
- Expand English contractions (e.g., *"can't" → "cannot"*)
- Normalize ordinal numbers (e.g., *"21st" → "twenty first"*)
- Remove URLs, HTML tags, mentions, and numbers
- Preserve hashtag words (removes `#` only)

### Emoji & Punctuation Handling
- Convert emojis to textual tokens (e.g., 😂 → `emoji_face_with_tears_of_joy`)
- Replace `!` and `?` with semantic tokens (`exclaim`, `question`)
- Reduce elongated words (e.g., *soooo* → *soo*)

### Token-Level Processing
- Tweet-aware tokenization
- Stopword removal (negations preserved)
- Negation marking (`not good` → `good_NEG`)
- Lemmatization using WordNet

---

## Data Augmentation

An optional **synonym replacement** technique is implemented using WordNet:

- Randomly replaces words with synonyms
- Controlled via probability parameter
- Applied only to the training set to reduce overfitting

This helps increase lexical diversity without changing semantic meaning.

---

## Model

The cleaned text is intended for use with an **LSTM-based neural network**, suitable for:

- Sequence modeling
- Capturing long-term dependencies in text
- Handling variable-length input sequences

The model architecture and training steps are implemented later in the notebook.

---

## Libraries & Tools

Key dependencies include:

- Python
- Pandas
- NLTK
- spaCy (`en_core_web_sm`)
- Emoji
- TensorFlow / Keras (for LSTM modeling)

Make sure all NLTK resources and spaCy models are downloaded before running locally.

---

## How to Run

1. Clone the repository or open the notebook in Kaggle
2. Ensure dataset files are available at the specified paths
3. Install required dependencies
4. Run the notebook top to bottom

---

## Notes

- The notebook emphasizes **text preprocessing quality**, which is critical for NLP performance.
- Designed for educational purposes as part of an NLP/LSTM assignment.
- Paths, hyperparameters, and augmentation probabilities can be easily modified.

---

## License

This project is intended for academic and educational use.
