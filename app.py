import streamlit as st

# ✅ Move set_page_config to the very first Streamlit command
st.set_page_config(page_title="🍽️ Recipe Recommendation System", 
                   page_icon="🥗", layout="wide")

# ✅ Now import other libraries
import pandas as pd
import joblib
import ast
import logging
import plotly.express as px

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
st.set_page_config(page_title="🍽️ Recipe Recommendation System", page_icon="🥗", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
        .big-font { font-size:24px !important; font-weight: bold; }
        .red-text { color: red; font-weight: bold; }
        .green-text { color: green; font-weight: bold; }
        .bold-text { font-weight: bold; }
        .block-container { padding-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔍 Filter Recipes")
min_rating = st.sidebar.slider("⭐ Minimum Rating", 1.0, 5.0, 3.0, 0.5)
max_calories = st.sidebar.slider("🔥 Max Calories", 50, 1000, 500, 50)

# Main UI
st.title("🍽️ AI-Powered Recipe Recommendation System")
st.markdown("<p class='big-font'>Discover the best recipes based on your ingredients! 🥗</p>", unsafe_allow_html=True)

# Input field for ingredients
ingredients = st.text_area("📝 Enter ingredients (comma-separated)", placeholder="e.g., tomato, cheese, onion")

if st.button("🔍 Get Recommendations"):
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
                    "recipe_id": int(row["recipe_id"]),
                    "recipe_name": row["recipe_name"],
                    "image_url": row["image_url"],
                    "aver_rate": float(row["aver_rate"]),
                    "review_nums": int(row["review_nums"]),
                    "calories": float(row["calories"]),
                    "fat": float(row["fat"]),
                    "carbohydrates": float(row["carbohydrates"]),
                    "protein": float(row["protein"]),
                    "cholesterol": float(row["cholesterol"]),
                    "sodium": float(row["sodium"]),
                    "fiber": float(row["fiber"]),
                    "ingredients_list": recipe_ingredients,
                    "available_ingredients": available,
                    "missing_ingredients": missing,
                    "similarity": round(1 - dist, 2),
                    "instructions": row.get("instructions", "No instructions available")
                }
                if rec["aver_rate"] >= min_rating and rec["calories"] <= max_calories:
                    recommendations.append(rec)
            except Exception as e:
                logger.error(f"Error processing recipe {row['recipe_name']}: {e}")

        # Display results
        if recommendations:
            st.subheader(f"🍳 {len(recommendations)} Best Recipe(s) Found")
            
            for recipe in recommendations:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(recipe["image_url"], width=220)
                
                with col2:
                    st.markdown(f"### 🍲 {recipe['recipe_name']}")
                    st.write(f"**⭐ Rating:** {recipe['aver_rate']} / 5 ({recipe['review_nums']} reviews)")
                    st.write(f"**🔥 Calories:** {recipe['calories']} kcal")
                    st.write(f"**🍽️ Similarity Score:** {recipe['similarity']}")

                    # Ingredients section
                    with st.expander("✅ Available Ingredients"):
                        st.write(", ".join(recipe["available_ingredients"]) if recipe["available_ingredients"] else "None")

                    with st.expander("❌ Missing Ingredients"):
                        st.write(", ".join(recipe["missing_ingredients"]) if recipe["missing_ingredients"] else "None")

                    # Nutritional Information as Chart
                    nutrition_df = pd.DataFrame({
                        "Nutrient": ["Fat", "Carbs", "Protein", "Cholesterol", "Sodium", "Fiber"],
                        "Amount": [recipe["fat"], recipe["carbohydrates"], recipe["protein"],
                                   recipe["cholesterol"], recipe["sodium"], recipe["fiber"]]
                    })
                    fig = px.bar(nutrition_df, x="Nutrient", y="Amount", title="📊 Nutritional Breakdown",
                                 color="Nutrient", text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)

                    # Recipe Instructions
                    with st.expander("📜 Recipe Instructions"):
                        st.write(recipe["instructions"])

                    # Save favorite recipes
                    if st.button(f"💖 Save '{recipe['recipe_name']}' to Favorites"):
                        with open("favorites.txt", "a") as f:
                            f.write(recipe["recipe_name"] + "\n")
                        st.success("Added to Favorites!")

                    st.write("---")
        else:
            st.warning("No matching recipes found. Try different ingredients.")
    else:
        st.warning("Please enter at least one ingredient.")
