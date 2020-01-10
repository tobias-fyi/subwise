"""
:: SubWise :: Subreddit Recommendation API ::

A simple back-end Flask API for recommending
the best subreddits to post your masterpieces.
"""

import os
import pickle
import pandas as pd

from flask import Flask, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# === Load in the pickled pre-trainees === #
# LabelEncoder
with open("assets/05_le.pkl", "rb") as p:
    le = pickle.load(p)

# chi2 selector
with open("assets/05_selector.pkl", "rb") as p:
    selector = pickle.load(p)

# Vectorizer
with open("assets/05_vocab.pkl", "rb") as p:
    vocab = pickle.load(p)

# Naive Bayes model
with open("assets/05_nb.pkl", "rb") as p:
    nb = pickle.load(p)


def predict(post: str, n: int = 5):
    """
    Create subreddit predictions.

    Parameters
    ----------
    post : string
        Selftext that needs a home.
    n    : integer
        The desired name of the output file,
        not including the '.pkl' extension.

    Returns
    -------
    Python dictionary formatted as follows:
        [{'subreddit': 'PLC', 'proba': 0.014454},
        {'subreddit': 'Rowing', 'proba': 0.005206}]
    """
    # Vectorize the post -> sparse doc-term matrix
    post_sparse = vocab.transform([post])

    # Feature selection
    post_select = selector.transform(post_sparse)

    # Generate predicted probabilities from trained model
    proba = nb.predict_proba(post_select)

    # Wrangle into correct format
    return (
        pd.DataFrame(proba, columns=[le.classes_])  # Classes as column names
        .T.reset_index()  # Transpose so column names become index
        .rename(columns={"level_0": "subreddit", 0: "proba"})  # Rename for aesthetics
        .sort_values(by="proba", ascending=False)  # Sort by probability
        .iloc[:n]  # n-top predictions to serve
        .to_json(orient="records")
    )


@app.route("/", methods=["POST"])
def rec():
    """
    Primary recommendation route.

    Parameters
    ----------
    post : string
        Content of post.
    n : int, optional
        Number of recommendations to return, by default 5.
    Returns
    -------
    top : JSON
        Returns a JSON array of top n recommendations and their
        relative probabilities.
    """

    req_data = request.json
    post = req_data["post"]
    n = req_data["n"]

    top = predict(post, n)

    return str(top)


if __name__ == "__main__":
    app.run()
