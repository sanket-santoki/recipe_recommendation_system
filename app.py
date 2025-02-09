import streamlit as st
import pandas as pd
import joblib
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained models and dataset
@st.cache_resource
def load_models():
    try:
        nn_model = joblib.load("recipe_nn_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        df = joblib.load("recipe_data.pkl")
        logger.info("Model and dataset loaded successfully.")
        return nn_model, vectorizer, df
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Function to process ingredients
def get_recipe_ingredients(ing):
    if isinstance(ing, list):
        return ing
    if isinstance(ing, str):
        try:
            parsed = ast.literal_eval(ing)
            return list(parsed) if isinstance(parsed, (list, tuple)) else [str(parsed)]
        except Exception as e:
            logger.warning(f"Parsing error: {e}")
            return [item.strip().strip("'\"") for item in ing.strip("[]").split(",") if item.strip()]
    return []

# Load models
nn_model, vectorizer, df = load_models()

# Streamlit UI
st.title("🍽️ Recipe Recommendation System")
st.write("Enter the ingredients you have, and get personalized recipe recommendations!")

# Input field for ingredients
ingredients = st.text_area("Enter ingredients (comma-separated)", placeholder="e.g., tomato, cheese, onion")

if st.button("Get Recommendations"):
    if ingredients.strip() and nn_model and vectorizer:
        # Process user input
        user_list = [s.strip().lower() for s in ingredients.split(",") if s.strip()]
        user_text = " ".join(user_list)
        user_set = set(user_list)

        try:
            # Transform input into vector
            user_vector = vectorizer.transform([user_text])
            distances, indices = nn_model.kneighbors(user_vector)
        except Exception as e:
            st.error(f"Error during model inference: {e}")
            st.stop()

        recommendations = []
        for idx, dist in zip(indices[0], distances[0]):
            row = df.iloc[idx]
            recipe_ingredients = list(get_recipe_ingredients(row["ingredients_list"]))
            recipe_set = set(s.strip().lower() for s in recipe_ingredients)
            available = list(recipe_set.intersection(user_set))
            missing = list(recipe_set - user_set)

            try:
                rec = {
                    "recipe_name": row["recipe_name"],
                    "image_url": row["image_url"],
                    "aver_rate": float(row["aver_rate"]),
                    "review_nums": int(row["review_nums"]),
                    "calories": float(row["calories"]),
                    "ingredients_list": recipe_ingredients,
                    "available_ingredients": available,
                    "missing_ingredients": missing,
                    "similarity": round(1 - dist, 2)
                }
                recommendations.append(rec)
            except Exception as e:
                logger.error(f"Error processing recipe {row['recipe_name']}: {e}")

        # Display results
        if recommendations:
            st.subheader("Recommended Recipes:")
            for recipe in recommendations:
                st.markdown(f"### 🍲 {recipe['recipe_name']}")
                st.image(recipe["image_url"], width=250)
                st.write(f"**Rating:** {recipe['aver_rate']} ⭐ ({recipe['review_nums']} reviews)")
                st.write(f"**Calories:** {recipe['calories']} kcal")
                st.write(f"**Ingredients Available:** {', '.join(recipe['available_ingredients'])}")
                st.write(f"**Missing Ingredients:** {', '.join(recipe['missing_ingredients'])}")
                st.write("---")
        else:
            st.warning("No matching recipes found. Try different ingredients.")
    else:
        st.warning("Please enter at least one ingredient.")
