class Scoring:
    def __init__(self):
        self.scores = {
            "youtube": {"title": 0.5, "description": 0.3, "thumbnail": 0.2},
            "x": {"text": 0.6, "image": 0.4},
            "image": {"image": 0.7, "text": 0.3},
            "other_link": {"text": 0.7, "image": 0.3},
            "clip": {"link": 1.0}
        }

    def get_score(self, category, element):
        return self.scores.get(category, {}).get(element, 0)

    def calculate_final_result_from_models(self, results):
        category_scores = {category: 0 for category in self.categories}
        for result in results:
            for category, probability in result:
                category_scores[category] += probability

        combined_category_scores = {
            "anime_group": category_scores["anime"] + category_scores["manga"] + category_scores["manhwa"],
            "jeu vidéo": category_scores["jeu vidéo"],
            "film": category_scores["film"],
            "série": category_scores["série"],
            "musique": category_scores["musique"],
            "sport": category_scores["sport"],
            "humour": category_scores["humour"],
            "politique": category_scores["politique"],
            "autre": category_scores["autre"],
        }
        final_category = max(combined_category_scores, key=combined_category_scores.get)
        final_probability = combined_category_scores[final_category]

        if final_probability > 0.92:
            return final_category
        elif final_probability > 0.55:
            return final_category if final_category == "anime_group" else None
        else:
            return None
