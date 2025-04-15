import configparser


def load_config():
    config = configparser.ConfigParser()
    config.read('config.cfg')
    return {
        'DISCORD_TOKEN': config.get('DEFAULT', 'DISCORD_TOKEN'),
        'YOUTUBE_API_KEY': config.get('DEFAULT', 'YOUTUBE_API_KEY'),
        'TESSERACT_CMD': config.get('DEFAULT', 'TESSERACT_CMD')
    }
