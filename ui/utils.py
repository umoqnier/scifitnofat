import json
import random

RECIPIES_FILE = "ui/data/recipes_db.json"
DOLLAR_PESO_RATE = 18.5


def load_mock_recipe():
    data = load_json(RECIPIES_FILE)
    recipe = data[0]
    return data, recipe


def load_json(file_name: str) -> dict:
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


def extract_ingredients(data: list[dict]) -> set[str]:
    result = set()
    for recipe in data:
        for ingredient in recipe["extendedIngredients"]:
            result.add(ingredient["name"])
    return result


def get_nutrient(recipe, name):
    """Helper to extract specific nutrient from the complex JSON list."""
    for nut in recipe["nutrition"]["nutrients"]:
        if nut["name"] == name:
            return f"{nut['amount']:.0f}{nut['unit']}"
    return "0"


def generate_weekly_plan(filtered_recipes, mode: int):
    """
    Generates a weekly plan using ONLY the provided list of filtered recipes.
    Classifies them into Breakfast/Main pools based on keywords.
    """
    days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    plan = {}

    if mode == 1:
        breakfast_pool = [
            r for r in filtered_recipes if r.get("meal_type") == "breakfast"
        ]
        lunch_pool = [r for r in filtered_recipes if r.get("meal_type") == "lunch"]
        dinner_pool = [r for r in filtered_recipes if r.get("meal_type") == "dinner"]

        if not breakfast_pool:
            breakfast_pool = filtered_recipes
        if not lunch_pool:
            lunch_pool = filtered_recipes
        if not dinner_pool:
            dinner_pool = lunch_pool
        main_pool_dict = {r["id"]: r for r in (lunch_pool + dinner_pool)}
        main_pool = list(main_pool_dict.values())
    else:
        # 1. Classify Recipes from the filtered list
        morning_keywords = {
            "morning meal",
            "brunch",
            "beverage",
            "breakfast",
            "drink",
        }
        main_keywords = {"lunch", "main course", "main dish", "dinner"}

        # Filter using set intersection
        breakfast_pool = [
            r
            for r in filtered_recipes
            if not morning_keywords.isdisjoint(set(r.get("dishTypes", [])))
        ]
        main_pool = [
            r
            for r in filtered_recipes
            if not main_keywords.isdisjoint(set(r.get("dishTypes", [])))
        ]

        # Fallback if filtered lists are too small (mock data safety)
        # If the filtered list has no breakfasts, we force it to use what is available
        if not breakfast_pool:
            breakfast_pool = filtered_recipes
        if not main_pool:
            main_pool = filtered_recipes

    # 2. Tracking for Anti-Repetition
    last_breakfast_id = None
    last_lunch_id = None
    last_dinner_id = None

    for day in days:
        daily_menu = []

        # --- SELECT BREAKFAST ---
        # Filter candidates: remove yesterday's breakfast
        b_candidates = [r for r in breakfast_pool if r["id"] != last_breakfast_id]
        # If we ran out of variety, reset pool
        if not b_candidates:
            b_candidates = breakfast_pool

        selected_b = random.choice(b_candidates)
        last_breakfast_id = selected_b["id"]

        # Create a display copy
        b_display = selected_b.copy()
        b_display["title"] = f"Desayuno: {selected_b['title']}"
        daily_menu.append(b_display)

        # --- SELECT LUNCH ---
        l_candidates = [r for r in main_pool if r["id"] != last_lunch_id]
        if not l_candidates:
            l_candidates = main_pool

        selected_l = random.choice(l_candidates)
        last_lunch_id = selected_l["id"]

        l_display = selected_l.copy()
        l_display["title"] = f"Comida: {selected_l['title']}"
        daily_menu.append(l_display)

        # --- SELECT DINNER ---
        # Try to avoid yesterday's dinner AND today's lunch
        d_candidates = [
            r
            for r in main_pool
            if r["id"] != last_dinner_id and r["id"] != selected_l["id"]
        ]
        if not d_candidates:
            d_candidates = main_pool

        selected_d = random.choice(d_candidates)
        last_dinner_id = selected_d["id"]

        d_display = selected_d.copy()
        d_display["title"] = f"Cena: {selected_d['title']}"
        daily_menu.append(d_display)

        # Save to plan
        plan[day] = daily_menu

    return plan


def filter_recipes_by_ids(all_recipes, target_ids):
    """
    Filters a list of recipes to return only those matching the given IDs.

    Args:
        all_recipes (list): List of recipe dictionaries (the full dataset).
        target_ids (list): List of recipe IDs (integers) to retrieve.

    Returns:
        list: List of recipe dictionaries matching the IDs.
    """
    # Convert target_ids to a set for faster lookup (O(1))
    target_id_set = set(target_ids)

    # Filter the recipes
    filtered_recipes = [
        recipe for recipe in all_recipes if recipe.get("id") in target_id_set
    ]

    return filtered_recipes


def get_weekly_ingredients(weekly_plan: dict, mode: int) -> set:
    """
    Extracts all ingredients from the weekly plan and counts occurrences.
    """
    ingredients = set()
    if mode == 1:
        for _, meals in weekly_plan.items():
            for meal in meals:
                ingredients.update(meal.get("missing", []))
        return ingredients
    elif mode == 0:
        for _, meals in weekly_plan.items():
            for meal in meals:
                ingredients.update(
                    [i.get("nameClean") for i in meal.get("extendedIngredients", [])]
                )
        return ingredients


def filter_recipes_pantry(all_recipes, target_ids, recipes_info):
    """Filters all recipes and attaches metadata (like meal_type) from recipes_info."""

    target_id_set = set(target_ids)

    # Build lookup by id_json
    info_by_id = {info["id_json"]: info for info in recipes_info}

    filtered_recipes = []

    for recipe in all_recipes:
        recipe_id = recipe.get("id")

        if recipe_id in target_id_set:
            # Copy the recipe to avoid modifying the original
            r = recipe.copy()

            # If this ID exists in the metadata info, merge it
            if recipe_id in info_by_id:
                r.update(
                    {
                        "meal_type": info_by_id[recipe_id]["meal_type"],
                        "matched": info_by_id[recipe_id]["matched"],
                        "missing": info_by_id[recipe_id]["missing"],
                        "missing_count": info_by_id[recipe_id]["missing_count"],
                    }
                )

            filtered_recipes.append(r)

    return filtered_recipes
