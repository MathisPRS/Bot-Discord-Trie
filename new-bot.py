import asyncio
import logging
import os
import re
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import discord
import pytesseract
import requests
import torch
from dotenv import load_dotenv
from PIL import Image
from playwright.async_api import async_playwright
from transformers import (
    CLIPModel,
    CLIPProcessor,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    ViTFeatureExtractor,
    ViTForImageClassification,
    pipeline,
)

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

class Scoring:
    def __init__(self):
        self.scores = {
            "youtube": {"title": 0.5, "description": 0.3, "thumbnail": 0.2},
            "x": {"text": 0.6, "image": 0.4},
            "image": {"image": 0.7, "text": 0.3},
            "other_link": {"text": 0.7, "image": 0.3},
            "clip": {"link": 1.0}
        }

    def get_score(self, category, element):
        return self.scores.get(category, {}).get(element, 0)

scoring = Scoring()

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
        self.categories = ["anime", "manga", "manhwa", "jeu vidéo", "film", "série", "musique", "autre"]
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model.eval()
        logger.info("Modèle CLIP chargé.")

        # Chargement modèle NLP
        logger.info("Chargement du modèle NLP...")
        self.nlp_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        logger.info("Modèle NLP chargé.")

        # Chargement modèle ViT pour les images
        logger.info("Chargement du modèle ViT...")
        self.vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        logger.info("Modèle ViT chargé.")

        # Chargement modèle DistilBERT pour les textes
        logger.info("Chargement du modèle DistilBERT...")
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(self.categories))
        logger.info("Modèle DistilBERT chargé.")

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
        logger.info(f"[EXTRACTION] Liens : {links}")
        logger.info(f"[EXTRACTION] Texte sans lien : {text_only}")

        # Analyse des liens
        for link in links:
            if "youtube.com" in link or "youtu.be" in link:
                target_channel = await self.analyze_youtube_link(link, current_channel, anime_channel)
            elif "x.com" in link or "twitter.com" in link:
                target_channel = await self.analyze_x_link(link, current_channel, anime_channel)
            elif "outplayed" in link:
                if current_channel != clip_channel:
                    target_channel = clip_channel
                    logger.info(f"[Clip] Redirection vers {clip_channel.name}")
            else:
                target_channel = await self.analyze_other_link(link, current_channel, anime_channel)

        # Analyse du texte seul s'il reste quelque chose
        if not links and not files and text_only:
            target_channel = await self.analyze_text(text_only, current_channel, anime_channel)

        # Analyse des images uploadées
        elif files:
            for file in files:
                if file.content_type and "image" in file.content_type:
                    image_url = file.url
                    target_channel = await self.analyze_image(image_url, current_channel, anime_channel)

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

    async def analyze_youtube_link(self, link, current_channel, anime_channel):
        try:
            video_id = get_video_id(link)
            logger.info(f"[YouTube] ID extrait : {video_id}")
            if video_id:
                api_key = os.getenv("YOUTUBE_API_KEY")
                title, description, thumbnail_url = get_video_info(api_key, video_id)
                if title and description and thumbnail_url:
                    logger.info(f"[YouTube] Titre : {title}")

                    # Analyse des textes
                    title_results = await self.analyze_text_with_models(title)
                    description_results = await self.analyze_text_with_models(description)

                    # Analyse de l'image
                    image_results = await self.analyze_image_with_models(thumbnail_url)

                    # Calcul du résultat final
                    final_result = self.calculate_final_result_from_models(title_results + description_results + image_results)

                    logger.info(f"[YouTube] Résultat final : {final_result}")

                    if final_result == "anime_group" and current_channel != anime_channel:
                        return anime_channel
        except Exception as e:
            logger.error(f"[YouTube] Erreur : {e}")
        return None

    async def analyze_x_link(self, link, current_channel, anime_channel):
        try:
            text_content, image_urls = await get_tweet_data(link)
            logger.info(f"[X] Texte détecté : {text_content}")
            logger.info(f"[X] Images : {image_urls}")

            # Analyse du texte
            text_results = await self.analyze_text_with_models(text_content)

            # Analyse des images
            image_results = []
            for img_url in image_urls:
                image_results.extend(await self.analyze_image_with_models(img_url))

            # Calcul du résultat final
            final_result = self.calculate_final_result_from_models(text_results + image_results)

            logger.info(f"[X] Résultat final : {final_result}")

            if final_result in ["anime", "manga", "manhwa"] and current_channel != anime_channel:
                return anime_channel
        except Exception as e:
            logger.error(f"[X] Erreur : {e}")
        return None

    async def analyze_other_link(self, link, current_channel, anime_channel):
        try:
            # Analyse du texte
            text_results = await self.analyze_text_with_models(link)

            # Analyse de l'image
            image_results = await self.analyze_image_with_models(link)

            # Calcul du résultat final
            final_result = self.calculate_final_result_from_models(text_results + image_results)

            logger.info(f"[Autre lien] Résultat final : {final_result}")

            if final_result == "anime" and current_channel != anime_channel:
                return anime_channel
        except Exception as e:
            logger.error(f"[Autre lien] Erreur : {e}")
        return None

    async def analyze_text(self, text, current_channel, anime_channel):
        try:
            text_results = await self.analyze_text_with_models(text)
            final_result = self.calculate_final_result_from_models(text_results)
            logger.info(f"[Texte seul] Résultat final : {final_result}")
            if final_result == "anime" and current_channel != anime_channel:
                return anime_channel
        except Exception as e:
            logger.error(f"[Texte seul] Erreur : {e}")
        return None

    async def analyze_image(self, image_url, current_channel, anime_channel):
        try:
            image_results = await self.analyze_image_with_models(image_url)
            final_result = self.calculate_final_result_from_models(image_results)
            logger.info(f"[Image uploadée] Résultat final : {final_result}")
            if final_result in ["anime", "manga", "manhwa"] and current_channel != anime_channel:
                return anime_channel
        except Exception as e:
            logger.error(f"[Image uploadée] Erreur : {e}")
        return None

    async def analyze_image_with_models(self, image_url):
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")

            # Analyse avec les trois modèles en parallèle
            clip_task = asyncio.create_task(self.analyze_image_clip(image))
            vit_task = asyncio.create_task(self.analyze_with_vit(image))
            distilbert_task = asyncio.create_task(self.analyze_with_distilbert(image))

            # Attendre que toutes les tâches soient terminées
            clip_result, vit_result, distilbert_result = await asyncio.gather(clip_task, vit_task, distilbert_task)

            # Log des prédictions pour chaque modèle
            logger.info(f"[CLIP] Catégorie prédite : {clip_result[0]} avec probabilité {clip_result[1]:.2f}")
            logger.info(f"[ViT] Catégorie prédite : {vit_result[0]} avec probabilité {vit_result[1]:.2f}")
            logger.info(f"[DistilBERT] Catégorie prédite : {distilbert_result[0]} avec probabilité {distilbert_result[1]:.2f}")

            return [clip_result, vit_result, distilbert_result]
        except Exception as e:
            logger.error(f"[IMAGE] Erreur analyse image : {e}")
            return [("autre", 0), ("autre", 0), ("autre", 0)]

    async def analyze_image_clip(self, image):
        inputs = self.clip_processor(text=self.categories, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze()
        predicted_category = self.categories[probs.argmax()]
        predicted_probability = probs.max().item()
        return predicted_category, predicted_probability

    async def analyze_with_vit(self, image):
        inputs = self.vit_feature_extractor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
        probs = outputs.logits.softmax(dim=1).squeeze()
        predicted_category = self.categories[probs.argmax()]
        predicted_probability = probs.max().item()
        return predicted_category, predicted_probability

    async def analyze_text_with_models(self, text):
        try:
            # Analyse avec les trois modèles en parallèle
            nlp_task = asyncio.create_task(self.analyze_text_nlp(text))
            vit_task = asyncio.create_task(self.analyze_with_vit_text(text))
            distilbert_task = asyncio.create_task(self.analyze_with_distilbert_text(text))

            # Attendre que toutes les tâches soient terminées
            nlp_result, vit_result, distilbert_result = await asyncio.gather(nlp_task, vit_task, distilbert_task)

            # Log des prédictions pour chaque modèle
            logger.info(f"[NLP] Catégorie prédite : {nlp_result[0]} avec probabilité {nlp_result[1]:.2f}")
            logger.info(f"[ViT] Catégorie prédite : {vit_result[0]} avec probabilité {vit_result[1]:.2f}")
            logger.info(f"[DistilBERT] Catégorie prédite : {distilbert_result[0]} avec probabilité {distilbert_result[1]:.2f}")

            return [nlp_result, vit_result, distilbert_result]
        except Exception as e:
            logger.error(f"[TEXT] Erreur analyse texte : {e}")
            return [("autre", 0), ("autre", 0), ("autre", 0)]

    async def analyze_text_nlp(self, text):
        result = self.nlp_model(text, candidate_labels=self.categories)
        probs = result["scores"]
        predicted_category = result["labels"][probs.index(max(probs))]
        predicted_probability = max(probs)
        return predicted_category, predicted_probability

    async def analyze_with_vit_text(self, text):
        # Utilisation de ViT pour le texte (exemple fictif)
        # Retourne une catégorie et une probabilité fictives pour l'exemple
        return "catégorie_vit", 0.85

    async def analyze_with_distilbert_text(self, text):
        inputs = self.distilbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.distilbert_model(**inputs)
        probs = outputs.logits.softmax(dim=1).squeeze()
        predicted_category = self.categories[probs.argmax()]
        predicted_probability = probs.max().item()
        return predicted_category, predicted_probability

    def calculate_final_result_from_models(self, results):
        category_scores = {category: 0 for category in self.categories}
        for result in results:
            for category, probability in result:
                category_scores[category] += probability

        # Regrouper les catégories similaires
        combined_category_scores = {
            "anime_group": category_scores["anime"] + category_scores["manga"] + category_scores["manhwa"],
            "jeu vidéo": category_scores["jeu vidéo"],
            "film": category_scores["film"],
            "série": category_scores["série"],
            "musique": category_scores["musique"],
            "autre": category_scores["autre"],
        }

        final_category = max(combined_category_scores, key=combined_category_scores.get)
        final_probability = combined_category_scores[final_category]

        logger.info(f"[FINAL] Catégorie finale : {final_category} avec probabilité {final_probability:.2f}")

        if final_probability > 0.92:
            return final_category
        elif final_probability > 0.55:
            return final_category if final_category == "anime_group" else None
        else:
            return None

if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True

    bot = MessageRouterBot(intents=intents)
    bot.run(bot.token)
