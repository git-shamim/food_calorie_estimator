# app/prompts_auto.py

# def get_classification_prompt():
#     return (
#         "What food item is shown in this image? "
#         "Describe the dish, its origin, typical ingredients, and whether it's homemade or restaurant-style."
#     )

# prompts_auto.py

def get_calorie_estimation_prompt(food_name: str) -> str:
    return (
        f"Estimate the total calories in one serving of '{food_name}'. "
        "Break down the calorie content by major ingredients. "
        "Provide the response in bullet points, without adding disclaimers."
    )

def get_health_evaluation_prompt(food_name: str) -> str:
    return (
        f"Evaluate the healthiness of the Indian dish '{food_name}'. "
        "Discuss its pros and cons from a nutritional perspective, "
        "considering fats, sugars, carbs, proteins, vitamins, and portion size. "
        "Respond in bullet points."
    )
