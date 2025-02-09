document.addEventListener('DOMContentLoaded', function() {
  const getRecBtn = document.getElementById('get-rec-btn');
  getRecBtn.addEventListener('click', function() {
    const ingredients = document.getElementById('ingredients').value.trim();
    if (!ingredients) {
      alert("Please enter some ingredients.");
      return;
    }
    fetchRecommendations(ingredients);
  });
});

function fetchRecommendations(ingredients) {
  const recipesContainer = document.getElementById('recipes-container');
  recipesContainer.innerHTML = "<p>Loading recommendations...</p>";
  
  fetch('/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ingredients: ingredients })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error("Network error: " + response.statusText);
    }
    return response.json();
  })
  .then(data => {
    displayRecipes(data);
  })
  .catch(error => {
    recipesContainer.innerHTML = `<p>Error: ${error.message}</p>`;
    console.error("Error fetching recommendations:", error);
  });
}

function displayRecipes(recipes) {
  const recipesContainer = document.getElementById('recipes-container');
  recipesContainer.innerHTML = "";
  
  if (!recipes || recipes.length === 0) {
    recipesContainer.innerHTML = "<p>No recommendations found.</p>";
    return;
  }
  
  recipes.forEach(recipe => {
    // Create the recipe card.
    const card = document.createElement('div');
    card.className = 'recipe-card';
    
    // Recipe image.
    const img = document.createElement('img');
    img.src = recipe.image_url;
    img.alt = recipe.recipe_name;
    card.appendChild(img);
    
    // Recipe information.
    const info = document.createElement('div');
    info.className = 'recipe-info';
    
    const title = document.createElement('h2');
    title.textContent = `${recipe.recipe_name} (Rating: ${recipe.aver_rate}, Reviews: ${recipe.review_nums})`;
    info.appendChild(title);
    
    const details = document.createElement('p');
    details.innerHTML = `<strong>Calories:</strong> ${recipe.calories}<br>
                         <strong>Macros:</strong> Fat: ${recipe.fat}g, Carbs: ${recipe.carbohydrates}g, Protein: ${recipe.protein}g`;
    info.appendChild(details);
    
    card.appendChild(info);
    
    // Chips for Available Ingredients.
    const availContainer = document.createElement('div');
    availContainer.className = 'chips-container';
    const availLabel = document.createElement('p');
    availLabel.innerHTML = "<strong>Available Ingredients:</strong>";
    availContainer.appendChild(availLabel);
    if (Array.isArray(recipe.available_ingredients) && recipe.available_ingredients.length > 0) {
      recipe.available_ingredients.forEach(ing => {
        const chip = document.createElement('span');
        chip.className = 'chip available';
        chip.textContent = ing;
        availContainer.appendChild(chip);
      });
    } else {
      const chip = document.createElement('span');
      chip.className = 'chip available';
      chip.textContent = "None";
      availContainer.appendChild(chip);
    }
    card.appendChild(availContainer);
    
    // Chips for Missing Ingredients.
    const missingContainer = document.createElement('div');
    missingContainer.className = 'chips-container';
    const missingLabel = document.createElement('p');
    missingLabel.innerHTML = "<strong>Missing Ingredients:</strong>";
    missingContainer.appendChild(missingLabel);
    if (Array.isArray(recipe.missing_ingredients) && recipe.missing_ingredients.length > 0) {
      recipe.missing_ingredients.forEach(ing => {
        const chip = document.createElement('span');
        chip.className = 'chip missing';
        chip.textContent = ing;
        missingContainer.appendChild(chip);
      });
    } else {
      const chip = document.createElement('span');
      chip.className = 'chip missing';
      chip.textContent = "None";
      missingContainer.appendChild(chip);
    }
    card.appendChild(missingContainer);
    
    // Similarity Score.
    const similarity = document.createElement('p');
    similarity.className = 'similarity';
    similarity.innerHTML = `<strong>Similarity Score:</strong> ${recipe.similarity.toFixed(2)}`;
    card.appendChild(similarity);
    
    recipesContainer.appendChild(card);
  });
}
