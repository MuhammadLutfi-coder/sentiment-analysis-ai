import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('IMDB Dataset.csv')

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review'])

y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

text = ["This movie is amazing"]
text_vec = vectorizer.transform(text)

print("Prediction:", model.predict(text_vec)[0])
user_input = input("Enter a review: ")
user_vec = vectorizer.transform([user_input])

print("Sentiment:", model.predict(user_vec)[0])