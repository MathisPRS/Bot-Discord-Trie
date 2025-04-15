import logging
import os


def setup_logging():
    os.makedirs("log", exist_ok=True)

    # Configuration du logger principal
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Supprimer les handlers existants pour éviter les doublons
    if logger.hasHandlers():
        logger.handlers.clear()

    # Ajout d'un handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Ajout d'un handler pour le fichier
    file_handler = logging.FileHandler("log/bot-discord.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Empêcher la propagation des logs vers les loggers parents
    logger.propagate = False
