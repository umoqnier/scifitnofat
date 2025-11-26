import json
import random
import numpy as np
import pandas as pd
from typing import Any

# ---------------------------
# NUEVA LISTA DE COMUNES (REDUCIDA)
# ---------------------------
COMMON_ASSUMED_INGREDIENTS = {
    "agua",
    "water",
    "sal",
    "salt",
    "pimienta",
    "pepper",
    "black pepper",
    "azúcar",
    "azucar",
    "sugar",
    "aceite",
    "oil",
    "olive oil",
    "vegetable oil",
    "canola oil",
    "harina",
    "flour",
    "vinagre",
    "vinegar",
    "comino",
    "cumin",
    "canela",
    "cinnamon",
}

# Penalizaciones diferenciadas
COMMON_MISSING_PENALTY = 1  # penalización muy baja para ingredientes "comunes"
NORMAL_MISSING_PENALTY = 20  # penalización fuerte para ingredientes no comunes


def load_and_prepare(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    data = json.loads(txt)
    df = pd.DataFrame(data)

    def extract_ingredient_names(nut):
        try:
            return set([i["name"].strip().lower() for i in nut.get("ingredients", [])])
        except Exception:
            return set()

    def get_nutrient_value(nut, name):
        try:
            for n in nut.get("nutrients", []):
                if n.get("name") == name:
                    return float(n.get("amount", 0) or 0)
            return 0.0
        except Exception:
            return 0.0

    df["ingredients_list"] = df["nutrition"].apply(extract_ingredient_names)
    df["calories"] = df["nutrition"].apply(lambda x: get_nutrient_value(x, "Calories"))
    df["protein"] = df["nutrition"].apply(lambda x: get_nutrient_value(x, "Protein"))
    df["carbs"] = df["nutrition"].apply(
        lambda x: get_nutrient_value(x, "Carbohydrates")
    )
    df["fat"] = df["nutrition"].apply(lambda x: get_nutrient_value(x, "Fat"))
    df["dishTypes"] = df.get("dishTypes", pd.Series([[]] * len(df))).apply(
        lambda x: [d.lower() for d in x] if isinstance(x, list) else []
    )
    return df


# ---------------------------
# NORMALIZACIÓN / Fuzzy core
# ---------------------------
def normalize_ingredient_core(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    modifiers = [
        "diced",
        "canned",
        "fresh",
        "dried",
        "breast",
        "ground",
        "skinless",
        "boneless",
        "pieces",
        "shredded",
        "cooked",
        "whole",
        "finely",
        "reduced fat",
        "low sodium",
        "organic",
        "large",
        "small",
        "medium",
        "powder",
        "minced",
    ]
    for m in modifiers:
        name = name.replace(m, "").strip()
    if len(name) > 3 and name.endswith("s"):
        name = name[:-1]
    return name


def analyze_recipe_for_pantry(recipe_ingredients: set[str], pantry_items: set[str]):
    """
    Devuelve:
      - matched: ingredientes encontrados en alacena
      - missing_common: faltantes que están en COMMON_ASSUMED_INGREDIENTS
      - missing_normal: faltantes que NO están en COMMON_ASSUMED_INGREDIENTS
      - missing_all: unión de ambos
    """
    pantry_cores = {normalize_ingredient_core(p) for p in pantry_items}
    matched = set()
    missing_common = set()
    missing_normal = set()

    for ing in recipe_ingredients:
        core = normalize_ingredient_core(ing)
        if not core:
            continue
        found = False
        for p in pantry_cores:
            if p and (p in core or core in p):
                matched.add(ing)
                found = True
                break
        if not found:
            # decidir si es "común" o "normal"
            if ing in COMMON_ASSUMED_INGREDIENTS or core in COMMON_ASSUMED_INGREDIENTS:
                missing_common.add(ing)
            else:
                missing_normal.add(ing)

    missing_all = missing_common | missing_normal
    return matched, missing_common, missing_normal, missing_all


# ---------------------------
# CLASIFICACIÓN USANDO dishTypes
# ---------------------------
def classify_by_dishtypes(dishtypes: list[str]) -> str:
    ds = set([d.lower() for d in dishtypes])
    breakfast_keys = {"breakfast", "morning", "morning meal", "merienda", "desayuno"}
    lunch_keys = {"lunch", "main course", "main", "almuerzo", "comida", "side dish"}
    dinner_keys = {"dinner", "supper", "cena"}
    if ds & breakfast_keys:
        return "breakfast"
    if ds & lunch_keys:
        return "lunch"
    if ds & dinner_keys:
        return "dinner"
    return "lunch"


# ---------------------------
# FITNESS / ERROR GLOBAL (con penalizaciones diferenciadas)
# ---------------------------
def compute_global_error(
    selected_rows: list[dict[str, Any]],
    target: dict[str, float],
    weight_cal: float,
    weight_pro: float,
    weight_carb: float,
    weight_fat: float,
    common_missing_penalty: float,
    normal_missing_penalty: float,
):
    tot_cal = sum(r["calories"] for r in selected_rows)
    tot_pro = sum(r["protein"] for r in selected_rows)
    tot_car = sum(r["carbs"] for r in selected_rows)
    tot_fat = sum(r["fat"] for r in selected_rows)

    err_macros = (
        abs(tot_cal - target["calories"]) * weight_cal
        + abs(tot_pro - target["protein"]) * weight_pro
        + abs(tot_car - target["carbs"]) * weight_carb
        + abs(tot_fat - target["fat"]) * weight_fat
    )

    # contar faltantes únicos de cada tipo
    missing_common_union = set()
    missing_normal_union = set()
    for r in selected_rows:
        missing_common_union.update(r.get("missing_common", set()))
        missing_normal_union.update(r.get("missing_normal", set()))

    miss_pen = (
        len(missing_common_union) * common_missing_penalty
        + len(missing_normal_union) * normal_missing_penalty
    )

    err = err_macros + miss_pen

    stats = {
        "calories": tot_cal,
        "protein": tot_pro,
        "carbs": tot_car,
        "fat": tot_fat,
        "missing_common_count": len(missing_common_union),
        "missing_normal_count": len(missing_normal_union),
        "missing_count": len(missing_common_union | missing_normal_union),
    }
    return err, stats


# ---------------------------
# GENETIC ALGORITHM (6 genes)
# chromosome layout: [b1_idx, b2_idx, l1_idx, l2_idx, d1_idx, d2_idx]
# ---------------------------
def run_ga_select_6(
    recipes_enriched: list[dict[str, Any]],
    breakfast_pool: list[int],
    lunch_pool: list[int],
    dinner_pool: list[int],
    target_macros: dict[str, float],
    pop_size: int = 200,
    generations: int = 300,
    mutation_rate: float = 0.12,
    elite_frac: float = 0.05,
    weight_cal: float = 1.0,
    weight_pro: float = 2.0,
    weight_carb: float = 0.5,
    weight_fat: float = 0.5,
    common_missing_penalty: float = COMMON_MISSING_PENALTY,
    normal_missing_penalty: float = NORMAL_MISSING_PENALTY,
    random_seed: int = None,
):
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    def sample_pool(pool, k):
        if len(pool) >= k:
            return random.sample(pool, k)
        elif len(pool) > 0:
            return list(np.random.choice(pool, size=k, replace=True))
        else:
            return [None] * k

    def make_random_chrom():
        chrom = []
        chrom += sample_pool(breakfast_pool, 2)
        chrom += sample_pool(lunch_pool, 2)
        chrom += sample_pool(dinner_pool, 2)
        return chrom

    population = [make_random_chrom() for _ in range(pop_size)]

    all_pool = breakfast_pool + lunch_pool + dinner_pool
    if not all_pool:
        raise RuntimeError("No hay recetas disponibles en los pools.")
    for chrom in population:
        for i in range(len(chrom)):
            if chrom[i] is None:
                chrom[i] = random.choice(all_pool)

    def fitness(chrom):
        selected_rows = [recipes_enriched[g] for g in chrom]
        err, _ = compute_global_error(
            selected_rows,
            target_macros,
            weight_cal,
            weight_pro,
            weight_carb,
            weight_fat,
            common_missing_penalty,
            normal_missing_penalty,
        )
        dup_penalty = 0
        if len(set(chrom)) != len(chrom):
            dup_penalty += 1000 * (len(chrom) - len(set(chrom)))
        final_err = err + dup_penalty
        # use exponential scaling for more interpretable fitness
        fit = np.exp(-final_err / 100.0)
        return fit, final_err

    def tournament_selection(pop, fits, k=3):
        ids = random.sample(range(len(pop)), k)
        best = max(ids, key=lambda i: fits[i][0])
        return pop[best]

    def crossover(p1, p2):
        child = p1.copy()
        for start in (0, 2, 4):
            if random.random() < 0.5:
                child[start : start + 2] = p2[start : start + 2]

        # repair duplicates using same-category pools when possible
        def repair(child):
            seen = set()
            for pos, gene in enumerate(child):
                if gene in seen:
                    pool = {
                        0: breakfast_pool,
                        1: breakfast_pool,
                        2: lunch_pool,
                        3: lunch_pool,
                        4: dinner_pool,
                        5: dinner_pool,
                    }[pos]
                    choices = [g for g in pool if g not in child]
                    if choices:
                        child[pos] = random.choice(choices)
                    else:
                        child[pos] = random.choice(pool)
                else:
                    seen.add(gene)
            return child

        return repair(child)

    def mutate(chrom):
        if random.random() < mutation_rate:
            cat = random.choice([0, 1, 2])
            pos = {0: [0, 1], 1: [2, 3], 2: [4, 5]}[cat]
            replace_pos = random.choice(pos)
            pool = {0: breakfast_pool, 1: lunch_pool, 2: dinner_pool}[cat]
            if pool:
                candidate = random.choice(pool)
                others = [g for g in pool if g not in chrom]
                if others:
                    candidate = random.choice(others)
                chrom[replace_pos] = candidate
        return chrom

    best_chrom = None
    best_fit = -1
    best_err = float("inf")
    elite_n = max(1, int(pop_size * elite_frac))

    for gen in range(generations):
        fits = [fitness(ch) for ch in population]
        fit_values = [f[0] for f in fits]
        err_values = [f[1] for f in fits]
        idx_best = int(np.argmax(fit_values))
        if fit_values[idx_best] > best_fit:
            best_fit = fit_values[idx_best]
            best_chrom = population[idx_best].copy()
            best_err = err_values[idx_best]

        sorted_idx = np.argsort(fit_values)[::-1]
        new_pop = [population[i].copy() for i in sorted_idx[:elite_n]]
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fits)
            p2 = tournament_selection(population, fits)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        population = new_pop

    best_selected = [recipes_enriched[g] for g in best_chrom]
    final_err, final_stats = compute_global_error(
        best_selected,
        target_macros,
        weight_cal,
        weight_pro,
        weight_carb,
        weight_fat,
        common_missing_penalty,
        normal_missing_penalty,
    )
    return {
        "chromosome": best_chrom,
        "selected": best_selected,
        "error": final_err,
        "stats": final_stats,
        "fitness": best_fit,
    }


# ---------------------------
# WRAPPER: preparar pools y correr GA (parámetros como argumentos)
# ---------------------------
def generar_plan_2_dias_ga(
    df: pd.DataFrame,
    user_data: dict,
    weight_cal: float = 1.0,
    weight_pro: float = 2.0,
    weight_carb: float = 0.5,
    weight_fat: float = 0.5,
    common_missing_penalty: float = COMMON_MISSING_PENALTY,
    normal_missing_penalty: float = NORMAL_MISSING_PENALTY,
    pop_size: int = 300,
    generations: int = 500,
    mutation_rate: float = 0.12,
    elite_frac: float = 0.06,
    random_seed: int = 42,
):
    """
    Ejecuta pipeline. Pesos y otros hiperparámetros son argumentos (Opción C).
    Pantry + metas se piden por consola. Si deseas que sean argumentos, lo cambio.
    """
    pantry_input = user_data["ingredients"]
    pantry_items = set([i.strip().lower() for i in pantry_input if i.strip()])
    daily_cal = float(user_data["kcal"])
    daily_prot = float(user_data["proteins"])
    daily_carb = float(user_data["carbs"])
    daily_fat = float(user_data["fat"])

    target_macros = {
        "calories": daily_cal*2,
        "protein": daily_prot*2,
        "carbs": daily_carb*2,
        "fat": daily_fat*2,
    }

    recipes_enriched = []
    for idx, row in df.iterrows():
        matched, missing_common, missing_normal, missing_all = (
            analyze_recipe_for_pantry(row["ingredients_list"], pantry_items)
        )
        recipes_enriched.append(
            {
                "index": idx,
                "id_json": row.get("id", None),
                "title": row.get("title", ""),
                "ingredients_list": row["ingredients_list"],
                "calories": row["calories"],
                "protein": row["protein"],
                "carbs": row["carbs"],
                "fat": row["fat"],
                "dishTypes": row.get("dishTypes", []),
                "meal_type": classify_by_dishtypes(row.get("dishTypes", [])),
                "matched": matched,
                "missing_common": missing_common,
                "missing_normal": missing_normal,
                "missing": missing_all,
                "missing_count": len(missing_all),
            }
        )

    breakfast_pool = [
        i for i, r in enumerate(recipes_enriched) if r["meal_type"] == "breakfast"
    ]
    lunch_pool = [
        i for i, r in enumerate(recipes_enriched) if r["meal_type"] == "lunch"
    ]
    dinner_pool = [
        i for i, r in enumerate(recipes_enriched) if r["meal_type"] == "dinner"
    ]

    def supplement_pool(pool, needed):
        if len(pool) >= needed:
            return pool
        candidates = sorted(
            range(len(recipes_enriched)),
            key=lambda i: (
                recipes_enriched[i]["missing_count"],
                abs(recipes_enriched[i]["calories"] - target_macros["calories"] / 3),
            ),
        )
        for c in candidates:
            if c not in pool:
                pool.append(c)
            if len(pool) >= needed:
                break
        return pool

    breakfast_pool = supplement_pool(breakfast_pool, 2)
    lunch_pool = supplement_pool(lunch_pool, 2)
    dinner_pool = supplement_pool(dinner_pool, 2)

    breakfast_pool = list(dict.fromkeys(breakfast_pool))
    lunch_pool = list(dict.fromkeys(lunch_pool))
    dinner_pool = list(dict.fromkeys(dinner_pool))

    if not (breakfast_pool and lunch_pool and dinner_pool):
        raise RuntimeError(
            "No hay suficientes recetas en las categorías. Revisa dataset o amplía COMMON_ASSUMED_INGREDIENTS."
        )

    ga_res = run_ga_select_6(
        recipes_enriched,
        breakfast_pool,
        lunch_pool,
        dinner_pool,
        target_macros,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        elite_frac=elite_frac,
        weight_cal=weight_cal,
        weight_pro=weight_pro,
        weight_carb=weight_carb,
        weight_fat=weight_fat,
        common_missing_penalty=common_missing_penalty,
        normal_missing_penalty=normal_missing_penalty,
        random_seed=random_seed,
    )

    sel = ga_res["selected"]
    chrom = ga_res["chromosome"]
    stats = ga_res["stats"]

    return {
        "selection": sel,
        "selection_ids": [r["id_json"] for r in sel],
        "chromosome": chrom,
        "stats": stats,
        "error": ga_res["error"],
        "fitness": ga_res["fitness"],
    }
