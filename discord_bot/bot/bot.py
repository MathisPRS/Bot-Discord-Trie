import logging
import discord
from bot.handlers.message_handler import handle_message
from config import load_config
from models.clip_model import CLIPModelWrapper
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
            "anime", "manga", "manhwa", "jeu vidéo", "musique", "sport", "humour", "politique", "autre"
        ]

        self.scoring = Scoring()

        self.clip_model = CLIPModelWrapper()

    async def on_ready(self):
        logging.info(f"Connecté en tant que {self.user}")

    async def on_message(self, message):
        # Récupération des canaux spécifiques au serveur du message
        self.anime_channel = discord.utils.get(message.guild.channels, name=self.ANIME_CHANNEL_NAME)
        self.clip_channel = discord.utils.get(message.guild.channels, name=self.CLIP_CHANNEL_NAME)
        await handle_message(self, message)