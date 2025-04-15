import logging

import discord
import models
from bot.handlers.message_handler import handle_message
from config import load_config
from models.bert_model import BERTModelWrapper
from models.clip_model import CLIPModelWrapper
from models.distilbert_model import DistilBERTModelWrapper
from models.efficientnet_model import EfficientNetModelWrapper
from models.resnet_model import ResNetModelWrapper
from models.roberta_model import RoBERTaModelWrapper
from utils.logging_utils import setup_logging
from utils.scoring import Scoring


class MessageRouterBot(discord.Client):
    def __init__(self, *, intents):
        super().__init__(intents=intents)
        self.config = load_config()
        self.token = self.config['DISCORD_TOKEN']

        self.ANIME_CHANNEL_NAME = "📺-les-mangas-séries-film"
        self.CLIP_CHANNEL_NAME = "📺-clip-stylés"

        setup_logging()

        self.categories = [
            "anime", "manga", "manhwa", "jeu vidéo", "film",
            "série", "musique", "sport", "humour", "politique", "autre"
        ]

        self.scoring = Scoring()

        self.clip_model = CLIPModelWrapper()
        self.efficientnet_model = EfficientNetModelWrapper()
        self.resnet_model = ResNetModelWrapper()
        self.bert_model = BERTModelWrapper(self.categories)
        self.roberta_model = RoBERTaModelWrapper(self.categories)
        self.distilbert_model = DistilBERTModelWrapper(self.categories)

    async def on_ready(self):
        logging.info(f"Connecté en tant que {self.user}")

    async def on_message(self, message):
        await handle_message(self, message)
