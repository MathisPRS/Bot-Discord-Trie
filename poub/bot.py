import os
import re
import discord
import requests
import json
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
import pytesseract
import logging
import os, certifi
from playwright.async_api import async_playwright

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Spécifie le chemin de Tesseract
load_dotenv()

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

# Configuration de l'API Twitter
print("Hello le certif est la :"+ certifi.where())
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

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

class MessageRouterBot(discord.Client):
    def __init__(self, *, intents):
        super().__init__(intents=intents)
        load_dotenv()
        self.token = os.getenv("DISCORD_TOKEN")

        # Charger les mots-clés depuis le fichier JSON
        with open('mots.json', 'r', encoding='utf-8') as file:
            keywords = json.load(file)
            self.anime_keywords = keywords["anime_keywords"]
            self.clip_keywords = keywords["clip_keywords"]

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
            video_id = self.extract_youtube_id(content)
            if video_id:
                try:
                    api_key = os.getenv("YOUTUBE_API_KEY")
                    endpoint = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}"
                    response = requests.get(endpoint)
                    data = response.json()
                    items = data.get("items", [])
                    if not items:
                        logging.warning(f"[YouTube] Aucune vidéo trouvée pour l'ID : {video_id}")
                        return
                    title = items[0]["snippet"]["title"]
                    logger.info(f"[YouTube] Titre détecté : {title}")
                    result = self.analyze_youtube_title(title)

                    if result == "anime" and current_channel != anime_channel:
                        target_channel = anime_channel
                except Exception as e:
                    logger.error(f"[YouTube Error] {e}")

        # Cas lien X
        elif "x.com" in content or "twitter.com" in content:
            text_content, image_urls = await get_tweet_data(content)
            logger.info(f"[X] Contenu détecté : {text_content}")
            logger.info(f"[X] Images détectées : {image_urls}")

            # Analyser les images
            image_result = False
            for img_url in image_urls:
                result = await self.analyze_image_clip(img_url)
                if result in ["anime", "manga", "manhwa", "série"]:
                    image_result = True
                    break

            # Analyser le texte
            text_result = any(word in text_content for word in self.anime_keywords)

            if image_result or text_result:
                if current_channel != anime_channel:
                    target_channel = anime_channel
                    logger.info(f"Contenu anime détecté, redirection vers {anime_channel.name}")

        # Cas lien outplayed
        elif any(word in content for word in self.clip_keywords):
            if current_channel != clip_channel:
                target_channel = clip_channel
                logger.info(f"Clip vidéo détecté, redirection vers {clip_channel.name}")

        # Cas autre lien
        elif "http" in content:
            if any(word in content for word in self.anime_keywords):
                if current_channel != anime_channel:
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

    def extract_youtube_id(self, url):
        match = re.search(r"(?:v=|youtu\\.be/)([A-Za-z0-9_-]{11})", url)
        return match.group(1) if match else None

    def analyze_youtube_title(self, title):
        title = title.lower()
        anime_terms = self.anime_keywords

        if any(term in title for term in anime_terms):
            return "anime"
        return "ignore"

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
            anime_terms = self.anime_keywords
            if any(term in text for term in anime_terms):
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
