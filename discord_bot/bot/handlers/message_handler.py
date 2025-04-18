import logging
import re
import pytesseract
from PIL import Image
from io import BytesIO
import requests
import discord

from config import DISCORD_TOKEN, YOUTUBE_API_KEY, TESSERACT_CMD
from utils.image_utils import analyze_image_with_models, extract_links, get_tweet_data, get_video_id, get_video_info
from utils.text_utils import analyze_text_with_models, analyze_text


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

        if not files and not links:
            logging.info("[IGNORED] Aucun lien ou image détecté, message ignoré.")
            return

        for link in links:
            if "youtube.com" in link or "youtu.be" in link:
                target_channel = await analyze_youtube_link(bot, link)
            elif "x.com" in link or "twitter.com" in link:
                target_channel = await analyze_x_link(bot, link)
            elif "outplayed" in link:
                if current_channel != bot.clip_channel:
                    target_channel = bot.clip_channel
            else:
                target_channel = await analyze_generic_link(bot, link)

        for file in files:
            if file.content_type and "image" in file.content_type:
                target_channel = await analyze_image(bot, file.url)

        if not links and not files and text_without_links:
            final_result = await process_text_and_images(bot, text=text_without_links)
            if final_result == "anime_group" and current_channel != bot.anime_channel:
                target_channel = bot.anime_channel

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


async def process_text_and_images(bot, text=None, image_urls=None):
    results = {}

    if text:
        text_results = await analyze_text_with_models(bot, text)
        adjusted_text = await analyze_text(text, dict(text_results)) if text_results else {}
        results["text"] = list(adjusted_text.items())

    if image_urls:
        image_results = []
        for img in image_urls:
            image_results.extend(await analyze_image_with_models(bot, img))
        results["image"] = image_results

    if not results:
        return None

    return bot.scoring.calculate_final_result_from_models(results)


async def analyze_youtube_link(bot, link):
    try:
        video_id = get_video_id(link)
        if not video_id:
            return None

        title, description, thumbnail_url = get_video_info(YOUTUBE_API_KEY, video_id)
        youtube_text = f"{title}\n{description}"

        final_result = await process_text_and_images(bot, text=youtube_text, image_urls=[thumbnail_url])
        logging.info(f"[YouTube] Résultat final : {final_result}")
        if final_result == "anime_group":
            return bot.anime_channel
    except Exception as e:
        logging.error(f"[YouTube] Erreur : {e}")
    return None


async def analyze_x_link(bot, link):
    try:
        text_content, image_urls = await get_tweet_data(link)
        final_result = await process_text_and_images(bot, text=text_content, image_urls=image_urls)
        logging.info(f"[X] Résultat final : {final_result}")
        if final_result == "anime_group":
            return bot.anime_channel
    except Exception as e:
        logging.error(f"[X] Erreur : {e}")
    return None


async def analyze_generic_link(bot, link):
    try:
        final_result = await process_text_and_images(bot, text=link, image_urls=[link])
        logging.info(f"[Lien générique] Résultat final : {final_result}")
        if final_result == "anime_group":
            return bot.anime_channel
    except Exception as e:
        logging.error(f"[Lien générique] Erreur : {e}")
    return None


async def analyze_image(bot, image_url):
    try:
        extracted_text = await extract_text_from_image(image_url)
        final_result = await process_text_and_images(bot, text=extracted_text, image_urls=[image_url])
        if final_result == "anime_group":
            return bot.anime_channel
    except Exception as e:
        logging.error(f"[Image] Erreur : {e}")
    return None



async def extract_text_from_image(image_url):
    try:
        logging.info(f"[EXTRACTION] Extraction depuis : {image_url}")
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logging.error(f"[OCR] Erreur extraction texte : {e}")
        return ""
