import asyncio
import logging
import json

# Charger les mots clés depuis le fichier JSON
with open('mots.json', 'r', encoding='utf-8') as f:
    keywords = json.load(f)

async def analyze_text_with_models(bot, text):
    try:
        if not text.strip():
            logging.info("Aucun texte à analyser.")
            return []

        logging.info(f"Analyzing text: {text}")
        roberta_result = await bot.roberta_model.analyze(text)

        if roberta_result[1] < 0.25:
            logging.info("Aucune catégorie avec une probabilité supérieure à 30%.")
            return []
        
        best_category, best_probability = max(roberta_result, key=lambda x: x[1])
        logging.info(f"[RoBERTa] Catégorie prédite : {best_category} avec probabilité {best_probability} ")

        return [roberta_result]
    except Exception as e:
        logging.error(f"[TEXT] Erreur analyse texte : {e}")
        return []

async def analyze_text(text, model_results):
    try:
        # Initialiser un dictionnaire pour stocker les probabilités ajustées
        adjusted_results = {category: 0 for category in model_results.keys()}

        # Parcourir les mots clés et ajuster les probabilités si un mot clé est trouvé
        for category, words in keywords.items():
            for word in words:
                if word.lower() in text.lower():
                    adjusted_results[category] = 1.0
                    break

        # Si aucun mot clé n'est trouvé, retourner les résultats du modèle
        if all(value == 0 for value in adjusted_results.values()):
            return model_results

        return adjusted_results
    except Exception as e:
        logging.error(f"[TEXT] Erreur analyse texte : {e}")
        return model_results