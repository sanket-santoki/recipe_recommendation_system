import streamlit as st
import pandas as pd
import joblib
import ast
import logging
import plotly.express as px

# ✅ Set Streamlit Page Configuration
st.set_page_config(page_title="🍽️ Recipe Finder", page_icon="🥗", layout="wide")

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

# ✅ Extract unique ingredients for suggestions
all_ingredients = set()
for ing_list in df["ingredients_list"]:
    try:
        parsed_ing = ast.literal_eval(ing_list)
        if isinstance(parsed_ing, list):
            all_ingredients.update(parsed_ing)
    except Exception as e:
        logger.warning("Failed to parse ingredient list: %s", e)

all_ingredients = sorted(all_ingredients)  # Sort for better user experience

# ✅ Sidebar for User Input
st.sidebar.header("🔍 Search for Recipes")

# 🔥 **Autocomplete Ingredient Selection**
selected_ingredients = st.sidebar.multiselect(
    "Enter ingredients (Start typing...)", 
    options=all_ingredients, 
    default=[]
)

# 🔍 Button to trigger recommendation
if st.sidebar.button("Find Recipes 🍽️"):
    if not selected_ingredients:
        st.sidebar.error("⚠️ Please select at least one ingredient.")
    else:
        # ✅ Preprocess user input
        user_text = " ".join(selected_ingredients)
        user_set = set(selected_ingredients)

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
            recipe_ingredients = list(ast.literal_eval(row["ingredients_list"]))
            recipe_set = set(recipe_ingredients)
            available = list(recipe_set.intersection(user_set))
            missing = list(recipe_set - user_set)

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
                    "similarity": round(1 - dist, 2),
                    "recipe_link": f"https://www.allrecipes.com/search?q={row['recipe_name'].replace(' ', '+')}"
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
                    
                    # ✅ Recipe Link
                    st.markdown(f"🔗 **[View Full Recipe]({recipe['recipe_link']})**", unsafe_allow_html=True)

                st.markdown("---")  # Separator

            # ✅ Pie Chart for Ingredient Contribution
            st.subheader("📊 Ingredient Contribution in Recommended Recipes")
            all_ing = []
            for recipe in recommendations[:5]:
                all_ing.extend(recipe["ingredients_list"])
            ing_df = pd.DataFrame({"Ingredient": all_ing})
            fig = px.pie(ing_df, names="Ingredient", title="Ingredient Contribution",
                         hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("😕 No recipes match your ingredients. Try different ones!")

# ✅ Footer
st.markdown("---")
st.write("© 2025 Sanket Santoki | Recipe Recommendation System 🍽️")
