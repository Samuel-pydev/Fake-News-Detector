import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# --- Config ---
DATA_DIR = "News_Data"
FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
HOLDOUT_SIZE = 2000
RANDOM_STATE = 42
MAX_FEATURES = 5000
TEST_SIZE = 0.2


def load_data(fake_path, true_path):

    """
    Load fake and real news CSVs, merge title and text into a single
    content column, and assign binary labels (0 = fake, 1 = real).
 
    Returns:
        fake_dataset (DataFrame): Fake articles with content and label.
        true_dataset (DataFrame): Real articles with content and label.
    """

    fake_df = pd.read_csv(fake_path)[['title', 'text']].copy()
    true_df = pd.read_csv(true_path)[['title', 'text']].copy()

    fake_df['content'] = fake_df['title'] + " " + fake_df['text']
    true_df['content'] = true_df['title'] + " " + true_df['text']

    fake_dataset = fake_df[['content']].copy()
    true_dataset = true_df[['content']].copy()

    fake_dataset['label'] = 0
    true_dataset['label'] = 1

    return fake_dataset, true_dataset


def build_holdout(fake_dataset, true_dataset):
    """
    Sample 2000 fake and 2000 real articles to form a holdout set.
    The holdout is never seen during training or testing  used only
    for final evaluation.
 
    Returns:
        holdout (DataFrame): 4000 articles held out from the main dataset.
    """

    holdout_fake = fake_dataset.sample(HOLDOUT_SIZE, random_state=RANDOM_STATE)
    holdout_true = true_dataset.sample(HOLDOUT_SIZE, random_state=RANDOM_STATE)
    holdout = pd.concat([holdout_fake, holdout_true])
    return holdout


def build_main_dataset(fake_dataset, true_dataset, holdout):
    """
    Combine fake and real datasets, remove holdout articles,
    and shuffle the result.
 
    Returns:
        df (DataFrame): Shuffled dataset ready for training and testing.
    """

    df = pd.concat([fake_dataset, true_dataset], axis=0)
    df = df.drop(holdout.index)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def preprocess(df, holdout):
    """
    Split the dataset into train/test sets and vectorize text using TF-IDF.
    The vectorizer is fit only on training data to prevent data leakage.
 
    Returns:
        X_train, X_test, y_train, y_test: Train/test splits.
        X_holdout, y_holdout: Vectorized holdout set.
        vectorizer: Fitted TfidfVectorizer instance.
    """

    X = df['content']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    X_holdout = vectorizer.transform(holdout['content'])
    y_holdout = holdout['label']

    return X_train, X_test, y_train, y_test, X_holdout, y_holdout, vectorizer


def train(X_train, y_train):
    """
    Train a Logistic Regression classifier on the training set.
 
    Returns:
        model: Fitted LogisticRegression instance.
    """

    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_train, y_train, X_test, y_test, X_holdout, y_holdout):
    """
    Print train, test, holdout accuracy scores and classification report.
    """

    train_acc = model.score(X_train, y_train) * 100
    test_acc = model.score(X_test, y_test) * 100
    holdout_acc = model.score(X_holdout, y_holdout) * 100
    y_pred = model.predict(X_test)

    print(f"\nTrained...")
    print("Classification Report: \n",classification_report(y_test, y_pred))
    print(f"\nTrain Accuracy:   {train_acc:.2f}%")
    print(f"Test Accuracy:    {test_acc:.2f}%")
    print(f"Holdout Accuracy: {holdout_acc:.2f}%")


def save(model, vectorizer):
    """
    Persist the trained model and vectorizer to disk using joblib.
    """

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("\nModel Saved ✔")


def main():
    """
    End-to-end pipeline: load data, build datasets, preprocess,
    train, evaluate, and save the model.
    """

    fake_dataset, true_dataset = load_data(FAKE_CSV, TRUE_CSV)
    holdout = build_holdout(fake_dataset, true_dataset)
    df = build_main_dataset(fake_dataset, true_dataset, holdout)

    X_train, X_test, y_train, y_test, X_holdout, y_holdout, vectorizer = preprocess(df, holdout)

    model = train(X_train, y_train)
    evaluate(model, X_train, y_train, X_test, y_test, X_holdout, y_holdout)
    save(model, vectorizer)


if __name__ == "__main__":
    main()