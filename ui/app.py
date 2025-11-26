import streamlit as st
from utils import (
    load_json,
    extract_ingredients,
    generate_weekly_plan,
    get_nutrient,
    RECIPIES_FILE,
    filter_recipes_by_ids,
)
from genetic_algorithm import (
    generar_plan_2_dias_ga,
    load_and_prepare,
    COMMON_MISSING_PENALTY,
    NORMAL_MISSING_PENALTY,
)
from greedy_algorithm import build_week_plan

db = load_json(RECIPIES_FILE)

st.set_page_config(page_title="SciFitNoFat", page_icon="🥑", layout="wide")


with st.sidebar:
    st.title("SciFitNoFat 👨🏼‍🍳👩‍🔬")
    st.header("Configuración ⚙️")
    st.subheader("Ingresa tus macro nutrientes 🧮")
    st.caption("Cumplamos tus metas diarias :material/favorite:")
    proteins_target = st.number_input("Proteinas (g)", value=150)
    fat_target = st.number_input("Grasa (g)", value=60)
    carbs_target = st.number_input("Carbohidratos (g)", value=200)
    kcal_target = st.number_input("Kcal", value=2000)
    budget = st.number_input("Presupuesto Semanal", icon="💸", value=1500)

    option_map = {
        0: ":material/chef_hat: Generar compras",
        1: ":material/kitchen: Usar alacena",
    }
    mode = st.radio(
        "Modo",
        options=option_map.keys(),
        format_func=lambda x: option_map[x],
        help="Puedes solicitar una lista de ingredientes a comprar para la semana (:material/chef_hat:) o ingresar los ingredientes que ya tienes en tu alacena/refri (:material/kitchen:)",
    )
    ingredients = []
    if mode == 1:
        ingredients = st.multiselect(
            "Ingredientes",
            options=extract_ingredients(db),
            format_func=lambda ingredient: ingredient.title(),
        )
    if st.button("Generar Plan Semanal", type="primary"):
        st.session_state["plan_generated"] = True
        user_data = {
            "budget": budget,
            "ingredients": ingredients,
            "kcal": kcal_target,
            "proteins": proteins_target,
            "carbs": carbs_target,
            "fat": fat_target,
        }
        # Here you would call your friend's function:
        # st.session_state['weekly_data'] = efficient_recipe_selector(ingredients, macros, budget)
        data = load_and_prepare(RECIPIES_FILE)
        if mode == 1:
            result = generar_plan_2_dias_ga(
                data,
                user_data,
                weight_cal=1.0,
                weight_pro=1.0,
                weight_carb=1.0,
                weight_fat=1.0,
                common_missing_penalty=COMMON_MISSING_PENALTY,
                normal_missing_penalty=NORMAL_MISSING_PENALTY,
                pop_size=300,
                generations=500,
                mutation_rate=0.12,
                elite_frac=0.06,
                random_seed=42,
            )
            recipes_ids = result["selection_ids"]
        else:
            recipes_ids, totals = build_week_plan(
                budget_week=user_data["budget"],
                cal_target_day=user_data["kcal"],
                p_target_day=user_data["proteins"],
                f_target_day=user_data["fat"],
                c_target_day=user_data["carbs"],
                max_repeats_per_recipe=2,
            )
        filtered_recipes = filter_recipes_by_ids(db, recipes_ids)
        st.session_state["weekly_data"] = generate_weekly_plan(filtered_recipes)


if "plan_generated" not in st.session_state:
    st.info(
        "👈 Por favor ingresa tus datos en la barra lateral y presiona 'Generar Plan'."
    )

    # Preview of the layout (Placeholder)
    # st.header("Vista Previa de Ingredientes")
    # Mocking your existing ingredient logic
    # st.write("Selecciona tus ingredientes (Simulado):")
    # st.multiselect("Ingredientes", ["Aguacate", "Pollo", "Arroz"], ["Aguacate"])
else:
    if mode == 1 and len(ingredients) == 0 and "plan_generated" in st.session_state:
        st.session_state["plan_generated"] = False
        st.warning("👈 Selecciona tus ingredientes primero")
    elif "plan_generated" in st.session_state:
        st.subheader("📅 Tu Menú Semanal")
        weekly_plan = st.session_state["weekly_data"]
        days = list(weekly_plan.keys())

        # Create Tabs for each day
        day_tabs = st.tabs(days)

        # Iterate through days and tabs
        for day, tab in zip(days, day_tabs):
            with tab:
                daily_recipes = weekly_plan[day]

                # 1. Daily Summary Header
                total_cals = sum(
                    [r["nutrition"]["nutrients"][0]["amount"] for r in daily_recipes]
                )
                total_price = (
                    sum([r["pricePerServing"] for r in daily_recipes]) / 100
                )  # Assuming price is in cents

                # Progress bars comparing to targets
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Costo del Día", f"${total_price:.2f}")
                c2.metric(
                    "Calorías Totales",
                    f"{total_cals:.0f}",
                    delta=f"{total_cals - kcal_target:.0f}",
                )
                c3.progress(
                    min(total_cals / kcal_target, 1.0), text="Progreso Calórico"
                )

                st.divider()

                # 2. Display Each Meal (Breakfast, Lunch, Dinner)
                for i, recipe in enumerate(daily_recipes):
                    # Helper container for visual grouping
                    with st.container():
                        st.markdown(f"### {recipe['title']}")

                        col_img, col_info = st.columns([1, 2])

                        with col_img:
                            st.image(recipe["image"], width="stretch")
                            # Display Tags
                            tags = [
                                diet.title() for diet in recipe.get("diets", [])[:3]
                            ]
                            st.caption(" • ".join(tags))

                        with col_info:
                            # Nutrient Metrics Row
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Kcal", get_nutrient(recipe, "Calories"))
                            m2.metric("Prot", get_nutrient(recipe, "Protein"))
                            m3.metric("Grasas", get_nutrient(recipe, "Fat"))
                            m4.metric("Carbs", get_nutrient(recipe, "Carbohydrates"))

                            st.markdown(
                                f"⏱️ **Tiempo:** {recipe['readyInMinutes']} mins | 💰 **Costo:** ${recipe['pricePerServing'] / 100:.2f}"
                            )

                            # Accordion for details (UX Best Practice: Progressive Disclosure)
                            with st.expander("📝 Ver Ingredientes y Pasos"):
                                ic1, ic2 = st.columns(2)
                                with ic1:
                                    st.markdown("**Ingredientes:**")
                                    for ing in recipe["extendedIngredients"]:
                                        st.markdown(f"- {ing['original']}")

                                with ic2:
                                    st.markdown("**Instrucciones:**")
                                    if recipe["analyzedInstructions"]:
                                        for step in recipe["analyzedInstructions"][0][
                                            "steps"
                                        ]:
                                            st.markdown(
                                                f"{step['number']}. {step['step']}"
                                            )
                                    else:
                                        st.write("No instructions available.")

                    if i < 2:  # Don't add divider after the last meal
                        st.divider()

        # --- SHOPPING LIST SUMMARY (Outside tabs) ---
        with st.expander("🛒 Ver Lista de Compras Semanal (Resumen)"):
            st.info("Aquí aparecería la suma de todos los ingredientes de la semana.")
