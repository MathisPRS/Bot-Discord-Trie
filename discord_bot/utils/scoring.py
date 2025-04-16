import logging

class Scoring:
    def __init__(self):
        self.scores = {
            "youtube": {"title": 0.5, "description": 0.3, "thumbnail": 0.2},
            "x": {"text": 0.6, "image": 0.4},
            "image": {"image": 1, "text": 0.5},
            "other_link": {"text": 0.7, "image": 0.3},
            "clip": {"link": 1.0}
        }
        self.categories = ["anime", "manga", "manhwa", "jeu vidéo", "film / série", "célébrité / influenceur", "musique", "sport / e-sport", "meme / humour", "nature / animaux", "autre"]

    def get_score(self, category, element):
        return self.scores.get(category, {}).get(element, 0)

    def calculate_final_result_from_models(self, results):
        category_scores = {category: 0 for category in self.categories}

        for result_type, result_list in results.items():
            for predicted_category, probability in result_list:
                category_scores[predicted_category] += probability * self.get_score("image", result_type)

        # Calcul des scores combinés
        combined_category_scores = {
            "anime_group": category_scores["anime"] + category_scores["manga"] + category_scores["manhwa"],
            "jeu vidéo": category_scores["jeu vidéo"],
            "film / série": category_scores["film / série"],
            "célébrité / influenceur": category_scores["célébrité / influenceur"],
            "musique": category_scores["musique"],
            "sport / e-sport": category_scores["sport / e-sport"],
            "meme / humour": category_scores["meme / humour"],
            "nature / animaux": category_scores["nature / animaux"],
            "autre": category_scores["autre"],
        }

        final_category = max(combined_category_scores, key=combined_category_scores.get)
        final_probability = combined_category_scores[final_category]
        logging.info(f"[FINAL RESULT] Score de la catégorie gagnante : {final_category} : {final_probability:.2f}")

        if final_probability > 0.92:
            return final_category
        elif final_probability > 0.55:
            return final_category if final_category == "anime_group" else None
        else:
            return None