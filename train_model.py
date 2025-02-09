import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import ast

def preprocess_ingredients(ing_str):
    """
    Convert a string representation of a list (e.g.,
    "['pork belly', 'smoked paprika', 'kosher salt', 'ground black pepper']")
    into a plain text string:
    "pork belly smoked paprika kosher salt ground black pepper"
    """
    try:
        ing_list = ast.literal_eval(ing_str)
        if isinstance(ing_list, list):
            return " ".join(ing_list)
        else:
            return ""
    except Exception:
        return ""

def train_nn_model():
    print("Loading cleaned data...")
    df = pd.read_csv("cleaned_recipe_data.csv")
    
    # Preprocess ingredients: create a plain-text version for TF-IDF.
    df['ingredients_text'] = df['ingredients_list'].apply(preprocess_ingredients)
    print("Example processed ingredients:", df.loc[0, 'ingredients_text'])
    
    # Build the TF-IDF vectorizer on the entire dataset.
    print("Vectorizing all ingredients...")
    vectorizer = TfidfVectorizer(lowercase=True)
    X = vectorizer.fit_transform(df['ingredients_text'])
    
    # Train a Nearest Neighbors model (n_neighbors=5) using cosine similarity.
    print("Training Nearest Neighbors model on full data...")
    nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn_model.fit(X)
    
    # Save the trained model, vectorizer, and full dataset.
    print("Saving model, vectorizer, and data...")
    joblib.dump(nn_model, "recipe_nn_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(df, "recipe_data.pkl")
    print("All components saved successfully.")

if __name__ == "__main__":
    train_nn_model()
