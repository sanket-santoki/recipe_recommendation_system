import streamlit as st
import pandas as pd
import joblib
import ast
import logging
import plotly.express as px

# ✅ Page Configuration (Mobile-Friendly)
st.set_page_config(page_title="🍽️ Recipe Finder", page_icon="🥗", layout="centered")

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load ML Artifacts
try:
    nn_model = joblib.load("recipe_nn_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    df = joblib.load("recipe_data.pkl")
    logger.info("Model artifacts loaded successfully.")
except Exception as e:
    logger.error("Error loading model artifacts: %s", e)
    st.error("⚠️ Failed to load model. Please check the files.")
    st.stop()

# ✅ Extract Unique Ingredients for Search Suggestions
all_ingredients = set()
for ing_list in df["ingredients_list"]:
    try:
        parsed_ing = ast.literal_eval(ing_list)
        if isinstance(parsed_ing, list):
            all_ingredients.update(parsed_ing)
    except Exception as e:
        logger.warning("Failed to parse ingredient list: %s", e)

all_ingredients = sorted(all_ingredients)  # Sort for better UX

# ✅ HOME PAGE UI
st.markdown(
    """
    <style>
        .title {text-align: center; font-size: 30px; font-weight: bold; margin-bottom: 20px;}
        .subtext {text-align: center; color: gray; margin-bottom: 30px;}
        .search-box {border-radius: 10px; padding: 10px; font-size: 16px;}
        .search-button {background-color: #ff4b4b; color: white; font-size: 16px; padding: 10px; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>🍽️ Recipe Finder</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Find the best recipes based on available ingredients!</div>", unsafe_allow_html=True)

# 🔥 **Autocomplete Ingredient Selection**
selected_ingredients = st.multiselect(
    "Enter ingredients (Start typing...)", 
    options=all_ingredients, 
    default=[]
)

# 🔍 Search Button
if st.button("Find Recipes 🍽️", key="search", help="Click to find recipes"):
    if not selected_ingredients:
        st.warning("⚠️ Please select at least one ingredient.")
    else:
        # ✅ Process Input
        user_text = " ".join(selected_ingredients)
        user_set = set(selected_ingredients)
        try:
            user_vector = vectorizer.transform([user_text])
            distances, indices = nn_model.kneighbors(user_vector)
        except Exception as e:
            logger.error("Error during model inference: %s", e)
            st.error("⚠️ Model inference failed.")
            st.stop()
        
        # ✅ Display Recipes
        recommendations = []
        for idx, dist in zip(indices[0], distances[0]):
            row = df.iloc[idx]
            recipe_ingredients = list(ast.literal_eval(row["ingredients_list"]))
            available = list(set(recipe_ingredients).intersection(user_set))
            missing = list(set(recipe_ingredients) - user_set)
            
            rec = {
                "recipe_name": row["recipe_name"],
                "image_url": row["image_url"],
                "calories": row["calories"],
                "protein": row["protein"],
                "carbohydrates": row["carbohydrates"],
                "fat": row["fat"],
                "available_ingredients": available,
                "missing_ingredients": missing,
                "recipe_link": f"https://www.allrecipes.com/search?q={row['recipe_name'].replace(' ', '+')}"
            }
            recommendations.append(rec)
        
        # ✅ UI for Displaying Recipes
        if recommendations:
            st.subheader("🍽️ Recommended Recipes")
            for recipe in recommendations[:5]:  # Show Top 5
                with st.container():
                    st.markdown(f"<h3>{recipe['recipe_name']}</h3>", unsafe_allow_html=True)
                    st.image(recipe["image_url"], width=300)
                    st.write(f"🔥 Calories: {recipe['calories']} kcal")
                    st.write(f"🍗 Protein: {recipe['protein']} g | 🥖 Carbs: {recipe['carbohydrates']} g | 🥑 Fat: {recipe['fat']} g")
                    st.write(f"✅ Available: {', '.join(recipe['available_ingredients'])}")
                    st.write(f"❌ Missing: {', '.join(recipe['missing_ingredients'])}")
                    st.markdown(f"[📖 View Full Recipe]({recipe['recipe_link']})", unsafe_allow_html=True)
                    st.markdown("---")
        else:
            st.warning("😕 No recipes found. Try different ingredients!")

# ✅ Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>© 2025 Sanket Santoki | Recipe Recommendation System 🍽️</p>", unsafe_allow_html=True)
