import logging
import ast
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Configure logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_recipe_ingredients(ing):
    """
    Convert the ingredients field to a list.
    If it's already a list, return it.
    If it's a string representation of a list, try ast.literal_eval.
    Otherwise, remove brackets and split by commas.
    """
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

# Load saved artifacts.
try:
    nn_model = joblib.load("recipe_nn_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    df = joblib.load("recipe_data.pkl")
    logger.info("Model artifacts loaded successfully.")
except Exception as e:
    logger.error("Error loading model artifacts: %s", e)
    raise e

@app.route("/favicon.ico")
def favicon():
    """Serve the favicon to prevent 404 errors."""
    favicon_path = os.path.join(app.root_path, "static")
    if os.path.exists(os.path.join(favicon_path, "favicon.ico")):
        return send_from_directory(favicon_path, "favicon.ico", mimetype="image/vnd.microsoft.icon")
    return "", 204

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True)
    if not data or "ingredients" not in data:
        return jsonify({"error": "No ingredients provided"}), 400

    user_input = data["ingredients"]
    if not user_input.strip():
        return jsonify({"error": "No ingredients provided"}), 400

    # Process the user input: split by comma, trim whitespace, and lowercase.
    user_list = [s.strip().lower() for s in user_input.split(",") if s.strip()]
    user_text = " ".join(user_list)
    user_set = set(user_list)

    try:
        user_vector = vectorizer.transform([user_text])
        distances, indices = nn_model.kneighbors(user_vector)
    except Exception as e:
        logger.error("Error during model inference: %s", e)
        return jsonify({"error": "Model inference failed"}), 500

    recommendations = []
    for idx, dist in zip(indices[0], distances[0]):
        row = df.iloc[idx]
        recipe_ingredients = list(get_recipe_ingredients(row["ingredients_list"]))
        recipe_set = set(s.strip().lower() for s in recipe_ingredients)
        available = list(recipe_set.intersection(user_set))
        missing = list(recipe_set - user_set)

        logger.info("Recipe: %s | Available: %s | Missing: %s",
                    row["recipe_name"], available, missing)

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
    
    return jsonify(recommendations)

if __name__ == "__main__":
    # For production, run behind a WSGI server (e.g., Gunicorn)
    app.run(host="0.0.0.0", port=5000, debug=True)
