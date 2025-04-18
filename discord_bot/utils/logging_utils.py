import os
import logging

def setup_logging():
    log_dir = "../log"
    log_file = os.path.join(log_dir, "bot-discord.log")

    # Créer le dossier de log s'il n'existe pas
    os.makedirs(log_dir, exist_ok=True)

    # Créer le fichier de log s'il n'existe pas
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('')  # On crée juste un fichier vide

    # Configuration du logger principal
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Supprimer les handlers existants pour éviter les doublons
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter commun
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler fichier
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Bloque la propagation
    logger.propagate = False
