import streamlit as st
import pandas as pd
import joblib
import ast
import logging
import plotly.express as px

# ✅ Set Streamlit Page Configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="🍽️ Recipe Recommendation System", 
                   page_icon="🥗", layout="wide")

# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load saved ML artifacts
try:
    nn_model = joblib.load("recipe_nn_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    df = joblib.load("recipe_data.pkl")
    logger.info("Model artifacts loaded successfully.")
except Exception as e:
    logger.error("Error loading model artifacts: %s", e)
    st.error("⚠️ Failed to load the model. Please check the files.")
    st.stop()

# ✅ Function to convert ingredients into a list
def get_recipe_ingredients(ing):
    if isinstance(ing, list):
        return ing
    if isinstance(ing, str):
        try:
            parsed = ast.literal_eval(ing)
            return list(parsed) if isinstance(parsed, (list, tuple)) else [str(parsed)]
        except Exception as e:
            logger.warning("literal_eval failed for input '%s': %s", ing, e)
            return [item.strip().strip("'\"") for item in ing.strip("[]").split(",") if item.strip()]
    return []

# ✅ Sidebar for User Input
st.sidebar.header("🔍 Search for Recipes")
ingredients = st.sidebar.text_input("Enter ingredients (comma-separated):", "")

# Optional filters
calories_limit = st.sidebar.slider("Max Calories", 50, 1000, 500)
protein_min = st.sidebar.slider("Min Protein (g)", 0, 100, 10)

# 🔍 Button to trigger recommendation
if st.sidebar.button("Find Recipes 🍽️"):
    if not ingredients.strip():
        st.sidebar.error("⚠️ Please enter at least one ingredient.")
    else:
        # ✅ Preprocess user input
        user_list = [s.strip().lower() for s in ingredients.split(",") if s.strip()]
        user_text = " ".join(user_list)
        user_set = set(user_list)

        try:
            user_vector = vectorizer.transform([user_text])
            distances, indices = nn_model.kneighbors(user_vector)
        except Exception as e:
            logger.error("Error during model inference: %s", e)
            st.error("⚠️ Model inference failed.")
            st.stop()

        recommendations = []
        for idx, dist in zip(indices[0], distances[0]):
            row = df.iloc[idx]
            recipe_ingredients = list(get_recipe_ingredients(row["ingredients_list"]))
            recipe_set = set(s.strip().lower() for s in recipe_ingredients)
            available = list(recipe_set.intersection(user_set))
            missing = list(recipe_set - user_set)

            # ✅ Apply filters
            if row["calories"] > calories_limit or row["protein"] < protein_min:
                continue

            try:
                rec = {
                    "recipe_id": int(row["recipe_id"]),
                    "recipe_name": row["recipe_name"],
                    "aver_rate": float(row["aver_rate"]),
                    "image_url": row["image_url"],
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
            except Exception as e:
                logger.error("Error constructing recipe for id %s: %s", row["recipe_id"], e)
                continue

            recommendations.append(rec)

        # ✅ Display Recommendations
        if recommendations:
            st.success(f"🎯 Found {len(recommendations)} matching recipes!")
            for recipe in recommendations[:5]:  # Show Top 5
                with st.container():
                    st.subheader(f"🍽️ {recipe['recipe_name']} (⭐ {recipe['aver_rate']})")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(recipe["image_url"], width=200)
                    with col2:
                        st.write(f"**📊 Similarity:** {recipe['similarity']*100:.1f}%")
                        st.write(f"**📌 Reviews:** {recipe['review_nums']} reviews")
                        st.write(f"🔥 **Calories:** {recipe['calories']} kcal")
                        st.write(f"🍗 **Protein:** {recipe['protein']} g")
                        st.write(f"🥖 **Carbs:** {recipe['carbohydrates']} g")
                        st.write(f"🥑 **Fat:** {recipe['fat']} g")
                        st.write(f"⚡ **Cholesterol:** {recipe['cholesterol']} mg")
                        st.write(f"🧂 **Sodium:** {recipe['sodium']} mg")
                        st.write(f"🌾 **Fiber:** {recipe['fiber']} g")

                    # ✅ Show Available & Missing Ingredients
                    st.markdown("✅ **Available Ingredients:** " + ", ".join(recipe["available_ingredients"]))
                    st.markdown("❌ **Missing Ingredients:** " + ", ".join(recipe["missing_ingredients"]))

                st.markdown("---")  # Separator

            # ✅ Interactive Nutritional Chart
            st.subheader("📊 Nutritional Comparison of Recommended Recipes")
            df_rec = pd.DataFrame(recommendations[:5])  # Convert to DataFrame
            fig = px.bar(df_rec, x="recipe_name", y=["calories", "protein", "carbohydrates", "fat"], 
                         title="Nutritional Breakdown", labels={"value": "Amount", "variable": "Nutrient"},
                         barmode="group", height=400)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("😕 No recipes match your filters. Try adjusting them.")

# ✅ Footer
st.markdown("---")
st.write("© 2025 Sanket Santoki | Recipe Recommendation System 🍽️")
