import json
import math
from collections import defaultdict

from utils import RECIPIES_FILE


def extract_macros(recipe):
    """Extrae calorías, proteína, grasa y carbohidratos."""
    nutrition = recipe.get("nutrition", {})
    nutrients = nutrition.get("nutrients", [])

    wanted = {
        "Calories": None,
        "Protein": None,
        "Fat": None,
        "Carbohydrates": None,
    }

    for n in nutrients:
        name = n.get("name")
        if name in wanted and wanted[name] is None:
            wanted[name] = n.get("amount")

    if any(v is None for v in wanted.values()):
        return (None, None, None, None)

    return (
        float(wanted["Calories"]),
        float(wanted["Protein"]),
        float(wanted["Fat"]),
        float(wanted["Carbohydrates"]),
    )


def get_recipes():
    with open(RECIPIES_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    recipes = []
    for r in raw_data:
        cal, prot, fat, carbs = extract_macros(r)
        if cal is None:
            continue

        price = r.get("pricePerServing")
        if price is None:
            continue
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue

        recipes.append(
            {
                "id": r.get("id"),
                "title": r.get("title", "Sin título"),
                "price": price,
                "cal": cal,
                "protein": prot,
                "fat": fat,
                "carbs": carbs,
            }
        )
    return recipes


# ============================================
# 3. Función objetivo (error macros/calorías semanal)
# ============================================
def objective(
    weekly_cal,
    weekly_prot,
    weekly_fat,
    weekly_carbs,
    cal_target_week,
    p_target_week,
    f_target_week,
    c_target_week,
    w_cal=1.0,
    w_p=1.0,
    w_f=1.0,
    w_c=1.0,
):
    return (
        w_cal * (weekly_cal - cal_target_week) ** 2
        + w_p * (weekly_prot - p_target_week) ** 2
        + w_f * (weekly_fat - f_target_week) ** 2
        + w_c * (weekly_carbs - c_target_week) ** 2
    )


# ============================================
# 4. Planner greedy para 21 comidas (7*3)
#    Devuelve: lista de ids (uno por comida) + summary
# ============================================
def plan_week_greedy(
    recipes,
    budget_week,
    cal_target_day,
    p_target_day,
    f_target_day,
    c_target_day,
    meals_target=21,
    max_repeats_per_recipe=2,  # default 2
    w_cal=1.0,
    w_p=1.0,
    w_f=1.0,
    w_c=1.0,
):
    # Targets semanales
    cal_target_week = cal_target_day * 7
    p_target_week = p_target_day * 7
    f_target_week = f_target_day * 7
    c_target_week = c_target_day * 7

    counts = defaultdict(int)  # id -> #porciones
    total_cost = 0.0
    total_meals = 0

    weekly_cal = 0.0
    weekly_prot = 0.0
    weekly_fat = 0.0
    weekly_carbs = 0.0

    current_obj = objective(
        weekly_cal,
        weekly_prot,
        weekly_fat,
        weekly_carbs,
        cal_target_week,
        p_target_week,
        f_target_week,
        c_target_week,
        w_cal,
        w_p,
        w_f,
        w_c,
    )

    id_to_recipe = {r["id"]: r for r in recipes}
    recipe_ids_sequence = []  # un id por comida

    while total_meals < meals_target:
        best_recipe = None
        best_new_obj = math.inf
        best_new_state = None

        for r in recipes:
            rid = r["id"]

            # Presupuesto
            new_cost = total_cost + r["price"]
            if new_cost > budget_week:
                continue

            # Máx repeticiones
            if counts[rid] >= max_repeats_per_recipe:
                continue

            # Macros nuevas
            new_cal = weekly_cal + r["cal"]
            new_prot = weekly_prot + r["protein"]
            new_fat = weekly_fat + r["fat"]
            new_carbs = weekly_carbs + r["carbs"]

            new_obj = objective(
                new_cal,
                new_prot,
                new_fat,
                new_carbs,
                cal_target_week,
                p_target_week,
                f_target_week,
                c_target_week,
                w_cal,
                w_p,
                w_f,
                w_c,
            )

            if new_obj < best_new_obj:
                best_new_obj = new_obj
                best_recipe = r
                best_new_state = (new_cost, new_cal, new_prot, new_fat, new_carbs)

        if best_recipe is None:
            print(
                "⚠️ No se pueden asignar más comidas sin violar el presupuesto o los límites."
            )
            break

        rid = best_recipe["id"]
        counts[rid] += 1
        total_meals += 1
        total_cost, weekly_cal, weekly_prot, weekly_fat, weekly_carbs = best_new_state
        current_obj = best_new_obj
        recipe_ids_sequence.append(rid)

    summary = {
        "meals_target": meals_target,
        "total_meals": total_meals,
        "total_cost": total_cost,
        "budget_week": budget_week,
        "weekly_cal": weekly_cal,
        "weekly_protein": weekly_prot,
        "weekly_fat": weekly_fat,
        "weekly_carbs": weekly_carbs,
        "cal_target_week": cal_target_week,
        "protein_target_week": p_target_week,
        "fat_target_week": f_target_week,
        "carbs_target_week": c_target_week,
        "cal_target_day": cal_target_day,
        "protein_target_day": p_target_day,
        "fat_target_day": f_target_day,
        "carbs_target_day": c_target_day,
        "objective_value": current_obj,
        "max_repeats_per_recipe": max_repeats_per_recipe,
    }

    return recipe_ids_sequence, summary


# ============================================
# 5. Función de alto nivel: build_week_plan
#    -> lista de ids, dict de totales
# ============================================
def build_week_plan(
    budget_week,
    cal_target_day,
    p_target_day,
    f_target_day,
    c_target_day,
    max_repeats_per_recipe=2,
    meals_target=21,
    w_cal=1.0,
    w_p=1.0,
    w_f=1.0,
    w_c=1.0,
):
    """
    Devuelve:
      - recipe_ids: [id1, id2, ..., idN] (uno por comida, puede haber repetidos)
      - totals: dict con los totales semanales y objetivos
    """
    recipes = get_recipes()
    recipe_ids, summary = plan_week_greedy(
        recipes=recipes,
        budget_week=budget_week,
        cal_target_day=cal_target_day,
        p_target_day=p_target_day,
        f_target_day=f_target_day,
        c_target_day=c_target_day,
        meals_target=meals_target,
        max_repeats_per_recipe=max_repeats_per_recipe,
        w_cal=w_cal,
        w_p=w_p,
        w_f=w_f,
        w_c=w_c,
    )

    totals = {
        "budget_week": summary["budget_week"],
        "total_cost": summary["total_cost"],
        "meals_target": summary["meals_target"],
        "total_meals": summary["total_meals"],
        "weekly_cal": summary["weekly_cal"],
        "weekly_protein": summary["weekly_protein"],
        "weekly_fat": summary["weekly_fat"],
        "weekly_carbs": summary["weekly_carbs"],
        "cal_target_week": summary["cal_target_week"],
        "protein_target_week": summary["protein_target_week"],
        "fat_target_week": summary["fat_target_week"],
        "carbs_target_week": summary["carbs_target_week"],
        "cal_target_day": summary["cal_target_day"],
        "protein_target_day": summary["protein_target_day"],
        "fat_target_day": summary["fat_target_day"],
        "carbs_target_day": summary["carbs_target_day"],
        "objective_value": summary["objective_value"],
        "max_repeats_per_recipe": summary["max_repeats_per_recipe"],
    }

    return recipe_ids, totals
