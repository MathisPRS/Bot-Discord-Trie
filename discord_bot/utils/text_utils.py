import asyncio
import logging


async def analyze_text_with_models(bot, text):
    try:
        logging.info(f"Analyzing text: {text}")
        bert_task = asyncio.create_task(bot.bert_model.analyze(text))
        roberta_task = asyncio.create_task(bot.roberta_model.analyze(text))
        distilbert_task = asyncio.create_task(bot.distilbert_model.analyze(text))

        bert_result, roberta_result, distilbert_result = await asyncio.gather(bert_task, roberta_task, distilbert_task)

        logging.info(f"[BERT] Catégorie prédite : {bert_result[0]} avec probabilité {bert_result[1]:.2f}")
        logging.info(f"[RoBERTa] Catégorie prédite : {roberta_result[0]} avec probabilité {roberta_result[1]:.2f}")
        logging.info(f"[DistilBERT] Catégorie prédite : {distilbert_result[0]} avec probabilité {distilbert_result[1]:.2f}")

        return [bert_result, roberta_result, distilbert_result]
    except Exception as e:
        logging.error(f"[TEXT] Erreur analyse texte : {e}")
        return [("autre", 0), ("autre", 0), ("autre", 0)]
