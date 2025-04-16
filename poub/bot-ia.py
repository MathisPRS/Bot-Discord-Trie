import discord
import requests
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
import pytesseract
import logging
import os
from playwright.async_api import async_playwright
from urllib.parse import urlparse, parse_qs
import re


# Création du dossier log s'il n'existe pas
os.makedirs("log", exist_ok=True)

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Ajout d'un handler fichier pour les logs
file_handler = logging.FileHandler("log/bot-discord.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
            logger.warning("[YouTube] Aucune vidéo trouvée avec cet ID.")
            return None, None, None
    else:
        logger.error(f"[YouTube] Erreur API : status {response.status_code}")
        return None, None, None
    

def extract_links(text):
    url_pattern = r'https?://[^\s]+'
    return re.findall(url_pattern, text)

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
        self.categories = ["anime", "manga", "manhwa", "jeu vidéo", "autre"]
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

        content = message.content
        links = extract_links(content)
        text_only = re.sub(r'https?://[^\s]+', '', content).strip()
        files = message.attachments
        current_channel = message.channel

        anime_channel = discord.utils.get(message.guild.channels, name=self.ANIME_CHANNEL_NAME)
        clip_channel = discord.utils.get(message.guild.channels, name=self.CLIP_CHANNEL_NAME)
        target_channel = None

        logger.info(f"[MESSAGE] Reçu dans #{current_channel.name} : {content}")

        # Analyse des liens
        for link in links:
            # Cas lien YouTube
            if "youtube.com" in link or "youtu.be" in link:
                try:
                    video_id = get_video_id(link)
                    logger.info(f"[YouTube] ID extrait : {video_id}")
                    if video_id:
                        api_key = os.getenv("YOUTUBE_API_KEY")
                        title, description, thumbnail_url = get_video_info(api_key, video_id)
                        if title and description and thumbnail_url:
                            logger.info(f"[YouTube] Titre : {title}")
                            title_result = self.analyze_text_nlp(title)
                            description_result = self.analyze_text_nlp(description)
                            image_result = await self.analyze_image_clip(thumbnail_url)

                            results = [title_result, description_result, image_result]
                            if results.count("anime") >= 2 and current_channel != anime_channel:
                                target_channel = anime_channel
                except Exception as e:
                    logger.error(f"[YouTube] Erreur : {e}")

            # Cas lien X / Twitter
            elif "x.com" in link or "twitter.com" in link:
                try:
                    text_content, image_urls = await get_tweet_data(link)
                    logger.info(f"[X] Texte détecté : {text_content}")
                    logger.info(f"[X] Images : {image_urls}")

                    image_result = "autre"
                    for img_url in image_urls:
                        result = await self.analyze_image_clip(img_url)
                        if result in ["anime", "manga", "manhwa"]:
                            image_result = result
                            break

                    text_result = self.analyze_text_nlp(text_content)
                    if (image_result in ["anime", "manga", "manhwa"] or text_result == "anime") and current_channel != anime_channel:
                        target_channel = anime_channel
                        logger.info(f"[X] Redirection vers {anime_channel.name}")
                except Exception as e:
                    logger.error(f"[X] Erreur : {e}")

            # Cas lien Outplayed
            elif "outplayed" in link:
                if current_channel != clip_channel:
                    target_channel = clip_channel
                    logger.info(f"[Clip] Redirection vers {clip_channel.name}")

            # Cas lien générique
            else:
                text_result = self.analyze_text_nlp(link)
                if text_result == "anime" and current_channel != anime_channel:
                    target_channel = anime_channel
                    logger.info(f"[Lien générique] Anime détecté, redirection vers {anime_channel.name}")

        # Analyse du texte seul s'il reste quelque chose
        if not links and not files and text_only:
            text_result = self.analyze_text_nlp(text_only)
            if text_result == "anime" and current_channel != anime_channel:
                target_channel = anime_channel
                logger.info(f"[Texte seul] Anime détecté, redirection vers {anime_channel.name}")

        # Analyse des images uploadées
        elif files:
            for file in files:
                if file.content_type and "image" in file.content_type:
                    image_url = file.url
                    result = await self.analyze_image_clip(image_url)
                    logger.info(f"[Image uploadée] Résultat CLIP : {result}")
                    if result in ["anime", "manga", "manhwa"] and current_channel != anime_channel:
                        target_channel = anime_channel

        # Redirection si nécessaire
        if target_channel:
            logger.info(f"[REDIRECTION] Vers #{target_channel.name}")
            await target_channel.send(
                f"🔁 Message de {message.author.mention} déplacé depuis {current_channel.mention} :\n{content}"
            )
            for file in files:
                file_data = await file.to_file()
                await target_channel.send(file=file_data)
            await message.delete()
        else:
            logger.info("Aucune redirection requise.")

    def analyze_text_nlp(self, text):
        try:
            result = self.nlp_model(text, candidate_labels=self.categories)
            predicted = result["labels"][0]
            logger.info(f"[NLP] Texte analysé : {text}")
            logger.info(f"[NLP] Catégorie prédite : {predicted}")
            return predicted
        except Exception as e:
            logger.error(f"[NLP] Erreur analyse texte : {e}")
            return "autre"

    async def analyze_image_clip(self, image_url):
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")

            # Analyse CLIP
            inputs = self.clip_processor(text=self.categories, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).squeeze()
            predicted_category = self.categories[probs.argmax()]

            logger.info(f"[CLIP] Catégorie prédite (image) : {predicted_category}")

            # Analyse OCR
            text = pytesseract.image_to_string(image).lower().strip()
            logger.info(f"[OCR] Texte détecté : {text}")

            if any(keyword in text for keyword in ["anime", "manga", "manhwa"]):
                logger.info("[OCR] Mot-clé anime détecté -> Catégorie forcée : anime")
                return "anime"

            return predicted_category
        except Exception as e:
            logger.error(f"[CLIP] Erreur analyse image : {e}")
            return "autre"

if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True

    bot = MessageRouterBot(intents=intents)
    bot.run(bot.token)
