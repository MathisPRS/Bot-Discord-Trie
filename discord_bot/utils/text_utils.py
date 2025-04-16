import asyncio
import logging

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
