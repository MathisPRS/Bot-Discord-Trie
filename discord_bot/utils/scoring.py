import logging

class Scoring:
    def __init__(self):
        self.scores = {
            "youtube": {"title": 0.5, "description": 0.3, "thumbnail": 0.2},
            "x": {"text": 0.6, "image": 0.4},
            "image": {"image": 1, "text": 0.6},
            "other_link": {"text": 0.7, "image": 0.3},
            "clip": {"link": 1.0}
        }
        self.categories = ["anime", "manga", "manhwa", "jeu vidéo", "film / série", "célébrité / influenceur", "musique", "sport / e-sport", "meme / humour", "nature / animaux", "autre"]

    def get_score(self, category, element):
        return self.scores.get(category, {}).get(element, 0)

    def calculate_final_result_from_models(self, results):
        try:
            # Étape 1 : Regrouper les scores texte et image dans deux dictionnaires séparés
            image_category_scores = {category: 0 for category in self.categories}
            text_category_scores = {category: 0 for category in self.categories}

            for result_type, result_list in results.items():
                for predicted_category, probability in result_list:
                    if result_type == "image":
                        image_category_scores[predicted_category] += probability * self.get_score("image", "image")
                    elif result_type == "text":
                        text_category_scores[predicted_category] += probability * self.get_score("image", "text")

            # Étape 2 : Regrouper les scores similaires dans chaque groupe
            def combine_scores(source_scores):
                return {
                    "anime_group": source_scores["anime"] + source_scores["manga"] + source_scores["manhwa"],
                    "jeu vidéo": source_scores["jeu vidéo"],
                    "film / série": source_scores["film / série"],
                    "célébrité / influenceur": source_scores["célébrité / influenceur"],
                    "musique": source_scores["musique"],
                    "sport / e-sport": source_scores["sport / e-sport"],
                    "meme / humour": source_scores["meme / humour"],
                    "nature / animaux": source_scores["nature / animaux"],
                    "autre": source_scores["autre"],
                }

            combined_image_scores = combine_scores(image_category_scores)
            combined_text_scores = combine_scores(text_category_scores)

            # Étape 3 : Trouver la meilleure catégorie pour chaque type
            best_image_category = max(combined_image_scores, key=combined_image_scores.get)
            best_image_score = combined_image_scores[best_image_category]

            best_text_category = max(combined_text_scores, key=combined_text_scores.get)
            best_text_score = combined_text_scores[best_text_category]

            logging.info(f"[SCORING] Image: {best_image_category} ({best_image_score:.2f}) | Texte: {best_text_category} ({best_text_score:.2f})")

            # Étape 4 : Choisir la catégorie finale
            if best_image_category == best_text_category:
                final_category = best_image_category
            else:
                final_category = best_image_category if best_image_score > best_text_score else best_text_category

            # Étape 5 : Vérification du seuil minimum
            if best_image_score < 0.3 and best_text_score < 0.3:
                logging.info("[FINAL RESULT] Scores trop faibles pour une redirection")
                return None

            logging.info(f"[FINAL RESULT] Catégorie retenue : {final_category}")
            return final_category

        except Exception as e:
            logging.error(f"[SCORING ERROR] Erreur pendant le calcul final : {e}")
            return None