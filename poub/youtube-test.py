import requests
from urllib.parse import urlparse, parse_qs

def get_video_id(url):
    # Extraire l'ID de la vidéo à partir de l'URL
    parsed_url = urlparse(url)
    video_id = parse_qs(parsed_url.query).get('v')
    if video_id:
        return video_id[0]
    else:
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

            print(f"Titre: {title}")
            print(f"Miniature: {thumbnail_url}")
            print(f"Description: {description}")
        else:
            print("Aucune information trouvée pour cette vidéo.")
    else:
        print(f"Erreur lors de la requête API: {response.status_code}")

# Exemple d'utilisation
api_key = "AIzaSyCJg1_5mqe1fGyddEu6pWxReZMxs_eh9uQ"
video_url = input("Veuillez entrer l'URL de la vidéo YouTube: ")

try:
    video_id = get_video_id(video_url)
    get_video_info(api_key, video_id)
except ValueError as e:
    print(e)
