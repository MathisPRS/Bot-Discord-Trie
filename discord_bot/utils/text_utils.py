import asyncio
import logging
import json
import re

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

        # Si c'est un seul tuple -> on l'emballe dans une liste
        if isinstance(roberta_result, tuple) and isinstance(roberta_result[1], float):
            roberta_result = [roberta_result]

        # Log du meilleur résultat
        best_category, best_probability = max(roberta_result, key=lambda x: x[1])
        logging.info(f"[RoBERTa] Catégorie prédite : {best_category} avec probabilité {best_probability}")

        return roberta_result
    except Exception as e:
        logging.error(f"[RoBERTa] Erreur analyse texte : {e}")
        return []

async def analyze_text(text, model_results):
    try:
        if not isinstance(model_results, dict):
            logging.error(f"[ANALYZE TEXT] model_results n'est pas un dictionnaire : {type(model_results)}")
            return model_results

        tokens = set(re.findall(r'\b\w+\b', text.lower()))
        anime_keywords = [kw.lower() for kw in keywords.get("anime_keywords", [])]

        for token in tokens:
            if token in anime_keywords:
                logging.info(f"[ANALYZE TEXT] Mot-clé anime détecté : {token}")
                model_results["anime"] = 1.0
                break  # Pas besoin de continuer après un match
        logging.info(f"[ANALYZE TEXT] Aucun mot-clé anime détecté")
        return model_results

    except Exception as e:
        logging.error(f"[ANALYZE TEXT] Erreur analyse texte : {e}")
        return model_results
