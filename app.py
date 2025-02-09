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

# Get unique ingredient list from dataset
ingredient_list = sorted(set(ing for sublist in df["ingredients_list"].dropna().apply(get_recipe_ingredients) for ing in sublist))

# Streamlit UI
st.set_page_config(page_title="Recipe Recommendation System", page_icon="🍽️", layout="wide")

st.title("🍽️ Recipe Recommendation System")
st.write("Enter the ingredients you have, and get personalized recipe recommendations!")

# Autocomplete dropdown for ingredients with auto-close feature
selected_ingredients = st.multiselect(
    "Select Ingredients:", 
    options=ingredient_list, 
    default=[],
    placeholder="Start typing to search ingredients..."
)

# Inject JavaScript to auto-close dropdown after selection
st.markdown(
    """
    <script>
        var dropdown = window.parent.document.querySelector("[data-testid='stMultiSelect']");
        if (dropdown) {
            dropdown.addEventListener("change", function() {
                setTimeout(() => {
                    dropdown.blur();
                }, 200);
            });
        }
    </script>
    """,
    unsafe_allow_html=True
)

if st.button("Get Recommendations"):
    if selected_ingredients and nn_model and vectorizer:
        # Process user input
        user_list = [s.strip().lower() for s in selected_ingredients]
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
                    "similarity": round(1 - dist, 2)
                }
                recommendations.append(rec)
            except Exception as e:
                logger.error(f"Error processing recipe {row['recipe_name']}: {e}")

        # Display results
        if recommendations:
            st.subheader("Recommended Recipes:")
            
            for recipe in recommendations:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(recipe["image_url"], width=200)
                
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
                    
                    # Nutritional Information
                    st.markdown("### 🥗 Nutritional Breakdown")
                    st.write(f"**Fat:** {recipe['fat']} g | **Carbohydrates:** {recipe['carbohydrates']} g")
                    st.write(f"**Protein:** {recipe['protein']} g | **Cholesterol:** {recipe['cholesterol']} mg")
                    st.write(f"**Sodium:** {recipe['sodium']} mg | **Fiber:** {recipe['fiber']} g")

                    st.write("---")
        else:
            st.warning("No matching recipes found. Try different ingredients.")
    else:
        st.warning("Please select at least one ingredient.")
