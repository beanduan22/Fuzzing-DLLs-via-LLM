import random

class APISetSelector:
    def __init__(self, grouped_apis, alpha):
        self.grouped_apis = grouped_apis
        self.alpha = alpha

    def calculate_api_scores(self):
        scores = {}
        for group, apis in self.grouped_apis.items():
            for api, usage in apis.items():
                scores[api] = 1 / (self.alpha * usage + 1)
        return scores

    def roulette_wheel_selection(self, api_scores, num_apis_to_select):
        total_score = sum(api_scores.values())
        probabilities = {api: score / total_score for api, score in api_scores.items()}

        selected_apis = set()
        while len(selected_apis) < num_apis_to_select:
            r = random.random()
            cumulative = 0
            for api, probability in probabilities.items():
                cumulative += probability
                if r <= cumulative:
                    selected_apis.add(api)
                    break
        return selected_apis

    def select_apis(self, total_apis_to_select):
        api_scores = self.calculate_api_scores()

        total_untouched = sum(len(apis) for apis in self.grouped_apis.values())
        group_selection_counts = {group: max(1, round(total_apis_to_select * len(apis) / total_untouched)) for group, apis in self.grouped_apis.items()}

        selected_apis = []
        for group, apis in self.grouped_apis.items():
            num_to_select = group_selection_counts[group]
            group_api_scores = {api: api_scores[api] for api in apis}
            selected_apis.extend(self.roulette_wheel_selection(group_api_scores, num_to_select))

        return selected_apis