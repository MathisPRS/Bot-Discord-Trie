import configparser

# Charger la configuration à partir du fichier config.cfg
config = configparser.ConfigParser()
config.read('config.cfg')

# Définir les variables de configuration
DISCORD_TOKEN = config.get('DEFAULT', 'DISCORD_TOKEN')
YOUTUBE_API_KEY = config.get('DEFAULT', 'YOUTUBE_API_KEY')
TESSERACT_CMD = config.get('DEFAULT', 'TESSERACT_CMD')
