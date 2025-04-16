import logging
import re
import pytesseract
from PIL import Image
from io import BytesIO
import requests
import discord
from config import DISCORD_TOKEN, YOUTUBE_API_KEY, TESSERACT_CMD
from utils.image_utils import analyze_image_with_models, extract_links, get_tweet_data, get_video_id, get_video_info
from utils.text_utils import analyze_text_with_models


async def handle_message(bot, message):
    if message.author.bot:
        logging.info("Message ignoré car envoyé par un bot.")
        return

    content = message.content.strip()
    links = extract_links(content)
    text_without_links = content
    for link in links:
        text_without_links = text_without_links.replace(link, "")

    files = message.attachments
    current_channel = message.channel

    target_channel = None

    if content or links or files:
        logging.info(f"[MESSAGE] Reçu dans #{current_channel.name} : {content}")
        if links:
            logging.info(f"[EXTRACTION] Liens : {links}")
        if text_without_links:
            logging.info(f"[EXTRACTION] Texte sans lien : {text_without_links}")

        if not message.attachments and not links:
            logging.info("[IGNORED] Aucun lien ou image détecté, message ignoré.")
            return

        # Analyse des liens
        for link in links:
            # Cas lien YouTube
            if "youtube.com" in link or "youtu.be" in link:
                target_channel = await analyze_youtube_link(bot, link)
            # Cas lien X / Twitter
            elif "x.com" in link or "twitter.com" in link:
                target_channel = await analyze_x_link(bot, link)
            # Cas lien Outplayed
            elif "outplayed" in link:
                if current_channel != bot.clip_channel:
                    target_channel = bot.clip_channel
            # Cas lien générique
            else:
                target_channel = await analyze_other_link(bot, link)

        # Analyse des images uploadées
        for file in files:
            if file.content_type and "image" in file.content_type:
                image_url = file.url
                target_channel = await analyze_image(bot, image_url)

        # Analyse du texte seul s'il reste quelque chose
        if not links and not files and text_without_links:
            text_results = await analyze_text_with_models(bot, text_without_links)
            final_result = bot.scoring.calculate_final_result_from_models(text_results)
            if final_result == "anime_group" and current_channel != bot.anime_channel:
                target_channel = bot.anime_channel

        # Redirection si nécessaire
        if target_channel and current_channel != target_channel:
            logging.info(f"[REDIRECTION] Vers #{target_channel.name}")
            await target_channel.send(
                f"🔁 Message de {message.author.mention} déplacé depuis {current_channel.mention} :\n{content}"
            )
            for file in files:
                file_data = await file.to_file()
                await target_channel.send(file=file_data)
            await message.delete()
        else:
            logging.info("Aucune redirection requise.")
    else:
        logging.info("Message vide ou sans contenu pertinent, ignoré.")

async def analyze_youtube_link(bot, link):
    try:
        video_id = get_video_id(link)
        logging.info(f"[YouTube] ID extrait : {video_id}")
        if video_id:
            api_key = YOUTUBE_API_KEY
            title, description, thumbnail_url = get_video_info(api_key, video_id)
            if title and description and thumbnail_url:
                logging.info(f"[YouTube] Titre : {title}")

                title_results = await analyze_text_with_models(bot, title)
                description_results = await analyze_text_with_models(bot, description)
                image_results = await analyze_image_with_models(bot, thumbnail_url)

                final_result = bot.scoring.calculate_final_result_from_models(title_results + description_results + image_results)

                logging.info(f"[YouTube] Résultat final : {final_result}")

                if final_result == "anime_group":
                    return bot.anime_channel
    except Exception as e:
        logging.error(f"[YouTube] Erreur : {e}")
    return None

async def analyze_x_link(bot, link):
    try:
        text_content, image_urls = await get_tweet_data(link)
        logging.info(f"[X] Texte détecté : {text_content}")
        logging.info(f"[X] Images : {image_urls}")

        text_results = await analyze_text_with_models(bot, text_content)

        image_results = []
        for img_url in image_urls:
            image_results.extend(await analyze_image_with_models(bot, img_url))

        final_result = bot.scoring.calculate_final_result_from_models(text_results + image_results)

        logging.info(f"[X] Résultat final : {final_result}")

        if final_result in ["anime", "manga", "manhwa"]:
            return bot.anime_channel
    except Exception as e:
        logging.error(f"[X] Erreur : {e}")
    return None

async def analyze_other_link(bot, link):
    try:
        text_results = await analyze_text_with_models(bot, link)
        image_results = await analyze_image_with_models(bot, link)

        final_result = bot.scoring.calculate_final_result_from_models(text_results + image_results)

        logging.info(f"[Autre lien] Résultat final : {final_result}")

        if final_result == "anime":
            return bot.anime_channel
    except Exception as e:
        logging.error(f"[Autre lien] Erreur : {e}")
    return None

async def analyze_image(bot, image_url):
    try:
        image_results = await analyze_image_with_models(bot, image_url)
        extracted_text = await extract_text_from_image(image_url)

        text_results = await analyze_text_with_models(bot, extracted_text)

        combined_results = {
            "image": image_results,
            "text": text_results
        }

        final_result = bot.scoring.calculate_final_result_from_models(combined_results)
        logging.info(f"[Image uploadée] Résultat final : {final_result}")
        if final_result == "anime_group":
            return bot.anime_channel
    except Exception as e:
        logging.error(f"[Image uploadée] Erreur : {e}")
    return None

async def extract_text_from_image(image_url):
    try:
        logging.info(f"Extracting text from image: {image_url}")
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        text = pytesseract.image_to_string(image)
        logging.info(f"Extracted text: {text}")
        return text.strip()
    except Exception as e:
        logging.error(f"[TEXT EXTRACTION] Erreur extraction texte : {e}")
        return ""