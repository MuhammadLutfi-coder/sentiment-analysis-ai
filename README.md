# Sentiment Analysis AI

This is a Machine Learning project that performs sentiment analysis on movie reviews using Python and Natural Language Processing (NLP).

---

## Project Overview

This AI model predicts whether a movie review is **Positive** or **Negative** based on text input.

It is trained using the IMDB dataset and uses traditional Machine Learning techniques.

---

## Dataset

- IMDB Movie Reviews Dataset
- Contains labeled reviews:
  - Positive
  - Negative

---

## Tech Stack

- Python 
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)

---

## Machine Learning Process

1. Load dataset
2. Text preprocessing
3. Feature extraction using CountVectorizer
4. Train model using Naive Bayes
5. Evaluate accuracy
6. Predict new user input

---

## Model Performance

- Accuracy: ~85% (varies based on split)

---

## Example Usage

Input: This movie really bad 

Output: Negative

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
