from litellm import model_cost
google_models = [m for m in model_cost.keys() if "gemini" in m.lower()]
print(google_models)
