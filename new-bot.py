import asyncio
import logging
import os
import re
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import pytesseract
import requests
import torch
from dotenv import load_dotenv
from PIL import Image
from playwright.async_api import async_playwright
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    CLIPModel,
    CLIPProcessor,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    EfficientNetForImageClassification,
    ResNetForImageClassification,
    RobertaForSequenceClassification,
    RobertaTokenizer,
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

class MessageRouterBot:
    def __init__(self):
        load_dotenv()

        # Chargement modèle CLIP pour les images
        logger.info("Chargement du modèle CLIP...")
        self.categories = ["anime", "manga", "manhwa", "jeu vidéo", "film", "série", "musique", "sport", "humour","politique" ,"autre"]
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model.eval()
        logger.info("Modèle CLIP chargé.")

        # Chargement modèle EfficientNet pour les images
        logger.info("Chargement du modèle EfficientNet...")
        self.efficientnet_model = EfficientNetForImageClassification.from_pretrained("efficientnet-b0")
        self.efficientnet_feature_extractor = pipeline("image-classification", model=self.efficientnet_model, feature_extractor="efficientnet-b0")
        logger.info("Modèle EfficientNet chargé.")
        
        # Chargement modèle ResNet pour les images
        logger.info("Chargement du modèle ResNet...")
        self.resnet_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.resnet_feature_extractor = pipeline("image-classification", model=self.resnet_model, feature_extractor="resnet-50")
        logger.info("Modèle ResNet chargé.")

        # Chargement modèle BERT pour les textes
        logger.info("Chargement du modèle BERT...")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(self.categories))
        logger.info("Modèle BERT chargé.")

        # Chargement modèle RoBERTa pour les textes
        logger.info("Chargement du modèle RoBERTa...")
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(self.categories))
        logger.info("Modèle RoBERTa chargé.")

        # Chargement modèle DistilBERT pour les textes
        logger.info("Chargement du modèle DistilBERT...")
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(self.categories))
        logger.info("Modèle DistilBERT chargé.")

    async def analyze_youtube_link(self, link):
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

                    if final_result == "anime_group":
                        return "anime_channel"
        except Exception as e:
            logger.error(f"[YouTube] Erreur : {e}")
        return None

    async def analyze_x_link(self, link):
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

            if final_result in ["anime", "manga", "manhwa"]:
                return "anime_channel"
        except Exception as e:
            logger.error(f"[X] Erreur : {e}")
        return None

    async def analyze_other_link(self, link):
        try:
            # Analyse du texte
            text_results = await self.analyze_text_with_models(link)

            # Analyse de l'image
            image_results = await self.analyze_image_with_models(link)

            # Calcul du résultat final
            final_result = self.calculate_final_result_from_models(text_results + image_results)

            logger.info(f"[Autre lien] Résultat final : {final_result}")

            if final_result == "anime":
                return "anime_channel"
        except Exception as e:
            logger.error(f"[Autre lien] Erreur : {e}")
        return None

    async def analyze_image(self, image_url):
        try:
            image_results = await self.analyze_image_with_models(image_url)
            final_result = self.calculate_final_result_from_models(image_results)
            logger.info(f"[Image uploadée] Résultat final : {final_result}")
            if final_result in ["anime", "manga", "manhwa"]:
                return "anime_channel"
        except Exception as e:
            logger.error(f"[Image uploadée] Erreur : {e}")
        return None

    async def analyze_image_with_models(self, image_url):
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")

            # Analyse avec les trois modèles en parallèle
            clip_task = asyncio.create_task(self.analyze_image_clip(image))
            efficientnet_task = asyncio.create_task(self.analyze_with_efficientnet(image))
            resnet_task = asyncio.create_task(self.analyze_with_resnet(image))

            # Attendre que toutes les tâches soient terminées
            clip_result, efficientnet_result, resnet_result = await asyncio.gather(clip_task, efficientnet_task, resnet_task)

            # Log des prédictions pour chaque modèle
            logger.info(f"[CLIP] Catégorie prédite : {clip_result[0]} avec probabilité {clip_result[1]:.2f}")
            logger.info(f"[EfficientNet] Catégorie prédite : {efficientnet_result[0]} avec probabilité {efficientnet_result[1]:.2f}")
            logger.info(f"[ResNet] Catégorie prédite : {resnet_result[0]} avec probabilité {resnet_result[1]:.2f}")

            return [clip_result, efficientnet_result, resnet_result]
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
        logger.info(f"[CLIP] Probabilités : {', '.join([f'{cat}: {probs[i]:.2f}' for i, cat in enumerate(self.categories)])}")
        return predicted_category, predicted_probability
    
    async def analyze_with_efficientnet(self, image):
        inputs = self.efficientnet_feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.efficientnet_model(**inputs)
        probs = outputs.logits.softmax(dim=1).squeeze()
        predicted_category = self.categories[probs.argmax()]
        predicted_probability = probs.max().item()
        logger.info(f"[EfficientNet] Probabilités : {', '.join([f'{cat}: {probs[i]:.2f}' for i, cat in enumerate(self.categories)])}")
        return predicted_category, predicted_probability
    
    async def analyze_with_resnet(self, image):
        inputs = self.resnet_feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.resnet_model(**inputs)
        probs = outputs.logits.softmax(dim=1).squeeze()
        predicted_category = self.categories[probs.argmax()]
        predicted_probability = probs.max().item()
        logger.info(f"[ResNet] Probabilités : {', '.join([f'{cat}: {probs[i]:.2f}' for i, cat in enumerate(self.categories)])}")
        return predicted_category, predicted_probability
    
    async def analyze_text_with_models(self, text):
        try:
            # Analyse avec les trois modèles en parallèle
            bert_task = asyncio.create_task(self.analyze_text_bert(text))
            roberta_task = asyncio.create_task(self.analyze_text_roberta(text))
            distilbert_task = asyncio.create_task(self.analyze_text_distilbert(text))

            # Attendre que toutes les tâches soient terminées
            bert_result, roberta_result, distilbert_result = await asyncio.gather(bert_task, roberta_task, distilbert_task)

            # Log des prédictions pour chaque modèle
            logger.info(f"[BERT] Catégorie prédite : {bert_result[0]} avec probabilité {bert_result[1]:.2f}")
            logger.info(f"[RoBERTa] Catégorie prédite : {roberta_result[0]} avec probabilité {roberta_result[1]:.2f}")
            logger.info(f"[DistilBERT] Catégorie prédite : {distilbert_result[0]} avec probabilité {distilbert_result[1]:.2f}")

            return [bert_result, roberta_result, distilbert_result]
        except Exception as e:
            logger.error(f"[TEXT] Erreur analyse texte : {e}")
            return [("autre", 0), ("autre", 0), ("autre", 0)]
        
    async def analyze_text_bert(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        probs = outputs.logits.softmax(dim=1).squeeze()
        predicted_category = self.categories[probs.argmax()]
        predicted_probability = probs.max().item()
        logger.info(f"[BERT] Probabilités : {', '.join([f'{cat}: {probs[i]:.2f}' for i, cat in enumerate(self.categories)])}")
        return predicted_category, predicted_probability
    
    async def analyze_text_roberta(self, text):
        inputs = self.roberta_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
        probs = outputs.logits.softmax(dim=1).squeeze()
        predicted_category = self.categories[probs.argmax()]
        predicted_probability = probs.max().item()
        logger.info(f"[RoBERTa] Probabilités : {', '.join([f'{cat}: {probs[i]:.2f}' for i, cat in enumerate(self.categories)])}")
        return predicted_category, predicted_probability
    
    async def analyze_text_distilbert(self, text):
        inputs = self.distilbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.distilbert_model(**inputs)
        probs = outputs.logits.softmax(dim=1).squeeze()
        predicted_category = self.categories[probs.argmax()]
        predicted_probability = probs.max().item()
        logger.info(f"[DistilBERT] Probabilités : {', '.join([f'{cat}: {probs[i]:.2f}' for i, cat in enumerate(self.categories)])}")
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
            "sport": category_scores["sport"],
            "humour": category_scores["humour"],
            "politique": category_scores["politique"],
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

async def process_message(bot, message_content):
    logger.info(f"Message reçu : {message_content}")
    links = extract_links(message_content)
    image_urls = re.findall(r'https?://[^\s]+\.(jpg|jpeg|png|gif)', message_content)

    target_channel = None

    # Analyse des liens
    for link in links:
        if "youtube.com" in link or "youtu.be" in link:
            target_channel = await bot.analyze_youtube_link(link)
        elif "x.com" in link or "twitter.com" in link:
            target_channel = await bot.analyze_x_link(link)
        elif "outplayed" in link:
            target_channel = "clip_channel"
        else:
            target_channel = await bot.analyze_other_link(link)

    # Analyse des images
    for image_url in image_urls:
        target_channel = await bot.analyze_image(image_url)

    if target_channel:
        logger.info(f"Message transféré dans le channel : {target_channel}")
    else:
        logger.info("Aucune redirection requise.")

async def main():
    bot = MessageRouterBot()
    while True:
        message_content = input("Entrez un message : ")
        await process_message(bot, message_content)

# Exécuter la boucle principale
asyncio.run(main())
