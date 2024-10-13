import numpy as np

class ContenderScorer:
    def __init__(self, contenders, scoring_strategy, **kwargs):
        """Initialize the ContenderScorer with a list of contenders."""
        self.contenders = contenders
        self.scoring_strategy = scoring_strategy
        self.kwargs = kwargs
        if scoring_strategy == 'quality_only':
            self.score_func = self.quality_only_score
        elif scoring_strategy == 'most_fair':
            self.score_func = self.most_fair_score
        elif scoring_strategy == 'balanced_tradeoff':
            self.score_func = self.balanced_tradeoff_score
        else:
            raise ValueError(f"Invalid scoring strategy '{scoring_strategy}'")

    def quality_only_score(self, contender):
        """Basic scoring strategy using quality score and speed factor."""
        return contender.quality_score * contender.speed_factor
    
    def most_fair_score(self, contender):
        """Scoring strategy based on fairness."""
        return -contender.total_requests_made # the contender with the least responses is the most fair
    
    def balanced_tradeoff_score(self, contender):
        """Balanced scoring strategy combining normalized quality, speed, and total requests with a tradeoff controlled by alpha."""
        # Normalize quality with log transformation (to reduce skew)
        normalized_quality = contender.quality_score

        # Normalize speed with min-max scaling (speed factor expected in range [0.5, 1.5])
        min_speed, max_speed = 0.0, 2.0
        normalized_speed = (contender.speed_factor - min_speed) / (max_speed - min_speed)
        normalized_speed = np.clip(normalized_speed, 0, 1)

        # UX component: combination of quality and speed
        ux_component = normalized_quality * normalized_speed

        # Normalize total requests relative to the fair share using log scaling
        total_requests = sum(c.total_requests_made for c in self.contenders)
        if total_requests > 0:
            fair_share = total_requests / len(self.contenders)
            normalized_requests = np.log1p(contender.total_requests_made / fair_share)
        else:
            # If no tasks have been made yet, set normalized_requests to 0
            normalized_requests = 0

        # Fairness component: penalizing highly requested contenders
        fairness_component = 1 / (1 + normalized_requests)

        # Final score: weighted combination of fairness and UX using alpha
        alpha = self.kwargs.get('alpha', None)
        score = (1 - alpha) * fairness_component + alpha * ux_component

        return score
    
    def calculate_scores(self):
        """Calculate scores for all contenders based on the selected scoring strategy."""
        scores = [self.score_func(c) for c in self.contenders]
        return scores
