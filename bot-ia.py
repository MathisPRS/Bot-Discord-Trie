import discord
import requests
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
import pytesseract
import logging
import os, certifi
from playwright.async_api import async_playwright
from urllib.parse import urlparse, parse_qs

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Spécifie le chemin de Tesseract
load_dotenv()

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

async def get_tweet_data(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        await page.wait_for_selector('//div[@data-testid="tweetText"]')

        try:
            # Extraire le texte du tweet
            tweet_text_element = await page.query_selector('//div[@data-testid="tweetText"]')
            tweet_text = await tweet_text_element.inner_text()

            # Extraire les images du tweet
            images = await page.query_selector_all('//img[contains(@src, "https://pbs.twimg.com/media/")]')
            image_urls = []
            if images:
                for img in images:
                    image_url = await img.get_attribute('src')
                    image_urls.append(image_url)

            return tweet_text, image_urls

        except Exception as e:
            logger.error(f"Erreur inattendue : {e}")
        finally:
            await browser.close()

def get_video_id(url):
    # Extraire l'ID de la vidéo à partir de l'URL
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
    logger.info(f"Requête API YouTube: {url}")
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        logger.info(f"Réponse API YouTube: {data}")
        if 'items' in data and len(data['items']) > 0:
            snippet = data['items'][0]['snippet']
            title = snippet['title']
            description = snippet['description']
            thumbnail_url = snippet['thumbnails']['high']['url']
            return title, description, thumbnail_url
        else:
            logger.warning("Aucune information trouvée pour cette vidéo.")
            return None, None, None
    else:
        logger.error(f"Erreur lors de la requête API: {response.status_code}")
        return None, None, None

class MessageRouterBot(discord.Client):
    def __init__(self, *, intents):
        super().__init__(intents=intents)
        load_dotenv()
        self.token = os.getenv("DISCORD_TOKEN")

        # Config des salons
        self.ANIME_CHANNEL_NAME = "📺-les-mangas-séries-film"
        self.CLIP_CHANNEL_NAME = "📺-clip-stylés"

        # Chargement modèle CLIP
        logger.info("Chargement du modèle CLIP...")
        self.categories = ["anime", "manga", "manhwa", "série", "jeu vidéo", "autre"]
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model.eval()
        logger.info("Modèle CLIP chargé.")

        # Chargement modèle NLP
        logger.info("Chargement du modèle NLP...")
        self.nlp_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        logger.info("Modèle NLP chargé.")

    async def on_ready(self):
        logger.info(f"Connecté en tant que {self.user}")

    async def on_message(self, message):
        if message.author.bot:
            return

        content = message.content.lower()
        files = message.attachments
        current_channel = message.channel

        anime_channel = discord.utils.get(message.guild.channels, name=self.ANIME_CHANNEL_NAME)
        clip_channel = discord.utils.get(message.guild.channels, name=self.CLIP_CHANNEL_NAME)
        target_channel = None

        logger.info(f"Message reçu dans le channel {current_channel.name} : {content}")

        # Cas lien YouTube
        if "youtube.com" in content or "youtu.be" in content:
            try:
                video_id = get_video_id(content)
                logger.info(f"ID de la vidéo YouTube extraite: {video_id}")
                if video_id:
                    api_key = os.getenv("YOUTUBE_API_KEY")
                    title, description, thumbnail_url = get_video_info(api_key, video_id)
                    if title and description and thumbnail_url:
                        logger.info(f"[YouTube] Titre détecté : {title}")

                        # Analyser le titre, la description et la miniature
                        title_result = self.analyze_text_nlp(title)
                        description_result = self.analyze_text_nlp(description)
                        image_result = await self.analyze_image_clip(thumbnail_url)

                        results = [title_result, description_result, image_result]
                        if results.count("anime") >= 2 and current_channel != anime_channel:
                            target_channel = anime_channel
            except Exception as e:
                logger.error(f"[YouTube Error] {e}")

        # Cas lien X
        elif "x.com" in content or "twitter.com" in content:
            text_content, image_urls = await get_tweet_data(content)
            logger.info(f"[X] Contenu détecté : {text_content}")
            logger.info(f"[X] Images détectées : {image_urls}")

            # Analyser les images
            image_result = "autre"
            for img_url in image_urls:
                result = await self.analyze_image_clip(img_url)
                if result in ["anime", "manga", "manhwa", "série"]:
                    image_result = result
                    break

            # Analyser le texte avec le modèle NLP
            text_result = self.analyze_text_nlp(text_content)

            if (image_result in ["anime", "manga", "manhwa", "série"] or text_result == "anime") and current_channel != anime_channel:
                target_channel = anime_channel
                logger.info(f"Contenu anime détecté, redirection vers {anime_channel.name}")

        # Cas lien outplayed
        elif "outplayed" in content:
            if current_channel != clip_channel:
                target_channel = clip_channel
                logger.info(f"Clip vidéo détecté, redirection vers {clip_channel.name}")

        # Cas autre lien
        elif "http" in content:
            text_result = self.analyze_text_nlp(content)
            if text_result == "anime" and current_channel != anime_channel:
                target_channel = anime_channel
                logger.info(f"Contenu anime détecté, redirection vers {anime_channel.name}")

        # Cas lien image ou upload
        elif files:
            for file in files:
                if file.content_type and "image" in file.content_type:
                    image_url = file.url
                    result = await self.analyze_image_clip(image_url)
                    logger.info(f"[CLIP] Résultat image : {result}")
                    if result in ["anime", "manga", "manhwa", "série"] and current_channel != anime_channel:
                        target_channel = anime_channel

        # Rediriger si besoin
        if target_channel:
            logger.info(f"Redirection du message vers le channel {target_channel.name}")
            await target_channel.send(
                f"🔁 Message de {message.author.mention} déplacé depuis {current_channel.mention} :\n{message.content}"
            )
            for file in files:
                file_data = await file.to_file()
                await target_channel.send(file=file_data)
            await message.delete()
        else:
            logger.info("Aucune redirection nécessaire.")

    def analyze_text_nlp(self, text):
        try:
            result = self.nlp_model(text, candidate_labels=self.categories)
            return result["labels"][0]
        except Exception as e:
            logger.error(f"[Erreur analyse texte] {e}")
            return "autre"

    async def analyze_image_clip(self, image_url):
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")

            # CLIP
            inputs = self.clip_processor(text=self.categories, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).squeeze()
            result = self.categories[probs.argmax()]

            # OCR
            text = pytesseract.image_to_string(image).lower()
            logger.info(f"[OCR] Texte détecté : {text}")
            if "anime" in text or "manga" in text or "manhwa" in text or "série" in text:
                return "anime"

            return result
        except Exception as e:
            logger.error(f"[Erreur analyse image] {e}")
            return "autre"

if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True

    bot = MessageRouterBot(intents=intents)
    bot.run(bot.token)
