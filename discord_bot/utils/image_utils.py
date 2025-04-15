import asyncio
import logging
import re
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import requests
from PIL import Image
from playwright.async_api import async_playwright


async def get_tweet_data(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        await page.wait_for_selector('//div[@data-testid="tweetText"]')

        try:
            logging.info(f"Extraction des données du tweet : {url}")
            tweet_text_element = await page.query_selector('//div[@data-testid="tweetText"]')
            tweet_text = await tweet_text_element.inner_text()

            images = await page.query_selector_all('//img[contains(@src, "https://pbs.twimg.com/media/")]')
            image_urls = []
            if images:
                for img in images:
                    image_url = await img.get_attribute('src')
                    image_urls.append(image_url)

            return tweet_text, image_urls

        except Exception as e:
            logging.error(f"Erreur inattendue : {e}")
        finally:
            await browser.close()

def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname in {'www.youtube.com', 'youtube.com', 'youtu.be'}:
        if parsed_url.path == '/watch':
            video_id = parse_qs(parsed_url.query).get('v')
            if video_id:
                return video_id[0]
        elif parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
    raise ValueError("URL invalide ou ID de vidéo non trouvé")

def get_video_info(api_key, video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if 'items' in data and len(data['items']) > 0:
            snippet = data['items'][0]['snippet']
            title = snippet['title']
            description = snippet['description']
            thumbnail_url = snippet['thumbnails']['high']['url']
            return title, description, thumbnail_url
        else:
            logging.warning("[YouTube] Aucune vidéo trouvée avec cet ID.")
            return None, None, None
    else:
        logging.error(f"[YouTube] Erreur API : status {response.status_code}")
        return None, None, None

def extract_links(text):
    url_pattern = r'https?://[^\s]+'
    return re.findall(url_pattern, text)

async def analyze_image_with_models(bot, image_url):
    try:
        logging.info(f"Analyzing image: {image_url}")
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        clip_task = asyncio.create_task(bot.clip_model.analyze(image, bot.categories))
        efficientnet_task = asyncio.create_task(bot.efficientnet_model.analyze(image, bot.categories))
        resnet_task = asyncio.create_task(bot.resnet_model.analyze(image, bot.categories))

        clip_result, efficientnet_result, resnet_result = await asyncio.gather(clip_task, efficientnet_task, resnet_task)

        logging.info(f"[CLIP] Catégorie prédite : {clip_result[0]} avec probabilité {clip_result[1]:.2f}")
        logging.info(f"[EfficientNet] Catégorie prédite : {efficientnet_result[0]} avec probabilité {efficientnet_result[1]:.2f}")
        logging.info(f"[ResNet] Catégorie prédite : {resnet_result[0]} avec probabilité {resnet_result[1]:.2f}")

        return [clip_result, efficientnet_result, resnet_result]
    except Exception as e:
        logging.error(f"[IMAGE] Erreur analyse image : {e}")
        return [("autre", 0), ("autre", 0), ("autre", 0)]
