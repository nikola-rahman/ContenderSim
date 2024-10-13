class Contender:
    def __init__(self, name, quality_score, speed_factor, response_time):
        self.name = name
        self.quality_score = quality_score
        self.speed_factor = speed_factor
        self.response_time = response_time
        self.total_requests_made = 0