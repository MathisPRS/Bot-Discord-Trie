import logging
from playwright.sync_api import sync_playwright
from PIL import Image
from io import BytesIO
import requests

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def get_tweet_data(tweet_url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(tweet_url)
        page.wait_for_selector('//div[@data-testid="tweetText"]')

        try:
            # Extraire le texte du tweet
            tweet_text_element = page.query_selector('//div[@data-testid="tweetText"]')
            tweet_text = tweet_text_element.inner_text()
            print("Texte du tweet :")
            print(tweet_text)
            print("\n")

            # Extraire les images du tweet
            images = page.query_selector_all('//img[contains(@src, "https://pbs.twimg.com/media/")]')
            if images:
                for img in images:
                    image_url = img.get_attribute('src')
                    print(f"URL de l'image : {image_url}")

                    # Télécharger et afficher l'image
                    img_response = requests.get(image_url)
                    img_data = Image.open(BytesIO(img_response.content))
                    img_data.show()
            else:
                print("Ce tweet ne contient pas d'images.")

        except Exception as e:
            logger.error(f"Erreur inattendue : {e}")
        finally:
            browser.close()

# Exemple d'utilisation
tweet_url = 'https://x.com/manele_dhm_/status/1910679061141107122'
get_tweet_data(tweet_url)
