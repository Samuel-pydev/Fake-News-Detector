import os
import joblib


# --- Config ---
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
DOCUMENT_PATH = "news.txt"
ENCODING = "utf-8"


def load():
    """
    Load the trained model and vectorizer from disk.

    Returns:
        model: Fitted LogisticRegression instance.
        vectorizer: Fitted TfidfVectorizer instance.
    """
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def read_doc(path):
    """
    Read a news article from a text file.

    Returns:
        news (str): Raw text content of the article.

    Raises:
        FileNotFoundError: If the document path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document not found: {path}")

    with open(path, "r", encoding=ENCODING) as f:
        news = f.read()

    return news


def predict(vectorizer, model, news):
    """
    Vectorize the input text and run the classifier.

    Returns:
        prediction (ndarray): Model prediction (0 = fake, 1 = real).
        X (sparse matrix): Vectorized input used for prediction.
    """
    X = vectorizer.transform([news])
    prediction = model.predict(X)

    if prediction[0] == 1:
        print("\n🖥: True news 📰")
    else:
        print("\n🖥: Fake news ❎")

    return prediction, X


def evaluate(model, X):
    """
    Print the model's confidence scores for both classes.
    """
    proba = model.predict_proba(X)[0]
    print(f"Fake confidence score: {proba[0] * 100:.2f}%")
    print(f"Real confidence score: {proba[1] * 100:.2f}%")


def main():
    """
    End-to-end prediction pipeline: load model, read article,
    predict label, and print confidence scores.
    """
    model, vectorizer = load()
    news = read_doc(DOCUMENT_PATH)
    prediction, X = predict(vectorizer, model, news)
    evaluate(model, X)


if __name__ == "__main__":
    main()