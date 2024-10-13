import argparse
import yaml
import pickle

import numpy as np

from contender import Contender
from scoring import ContenderScorer
from selector import ContenderSelector

def calc_fairness_score(contenders):
    def calculate_fair_distribution(contenders):
        quality_scores = np.array([c.quality_score*c.speed_factor for c in contenders])
        total_quality = np.sum(quality_scores)
        fair_distribution = quality_scores / total_quality
        return fair_distribution
    
    fair_distribution = calculate_fair_distribution(contenders)
    actual_distribution = np.array([c.total_requests_made for c in contenders])
    actual_distribution = actual_distribution / np.sum(actual_distribution)

    return 0.5 * np.sum(np.abs(actual_distribution - fair_distribution))

def calc_ux_score(speed_factor, quality_score, contenders):
    # Calculate the current score for the contender
    current_score = speed_factor * quality_score
    
    # Find the best possible score among all contenders
    best_possible_score = max(c.speed_factor * c.quality_score for c in contenders)
    
    # Normalize the current score relative to the best possible score
    normalized_ux_score = current_score / best_possible_score if best_possible_score > 0 else 0
    
    return normalized_ux_score

class Experiment:
    def __init__(self, config):
        self.config = config
        self.create_contenders()
        self.create_scorer()
        self.create_selector()

    def create_contenders(self):
        n_contenders = self.config["n_contenders"]
        
        self.contenders = [
            Contender(
                name=str(i),
                quality_score=np.random.beta(40, 2)**4,  # This gives a left-skewed distribution between 0 and 1
                speed_factor = np.clip(np.random.normal(1, 0.3), 0, 2),  # Gaussian around mean 1
                response_time=np.random.normal(1, 0.5)
            ) for i in range(n_contenders)
        ]

    def create_scorer(self):
        strategy = self.config.get("scorer", None).get("strategy", None)
        kwargs = self.config.get("scorer", None).get("kwargs", {})

        self.scorer = ContenderScorer(self.contenders, strategy, **kwargs)

    def create_selector(self):
        strategy = self.config.get("selector", None).get("strategy", None)
        kwargs = self.config.get("selector", None).get("kwargs", {})

        self.selector = ContenderSelector(strategy, self.contenders, self.scorer, **kwargs)

    def run(self, n_iterations):
        history = []

        for i in range(n_iterations):
            top_contender_idx = self.selector.select()
            top_contender = self.contenders[top_contender_idx]

            top_contender.total_requests_made += 1

            fairness_score = calc_fairness_score(self.contenders)
            ux_score = calc_ux_score(top_contender.speed_factor, top_contender.quality_score, self.contenders)

            history.append({
                "iteration": i,
                "top_contender": top_contender,
                "fairness_score": fairness_score,
                "ux_score": ux_score
            })

        return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate tasks for contenders using different strategies")
    parser.add_argument("--n_requests", type=int, help="Number of iterations to simulate")
    parser.add_argument("--n_contenders", type=int, help="Number of contenders to simulate")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # check if overwritten
    if args.n_contenders:
        print("Warning: Overwriting n_contenders from config file.")
        config["n_contenders"] = args.n_contenders

    if args.n_requests:
        print("Warning: Overwriting n_requests from config file.")
        config["n_requests"] = args.n_requests

    # iterate experiments in config 
    results = []
    for exp_config in config["experiments"]:
        exp_name = exp_config["name"]
        print(f"Running experiment: {exp_name}")
        exp_config.update(config["defaults"])
        experiment = Experiment(exp_config)

        history = experiment.run(exp_config["n_requests"])

        results.append({
            "experiment": exp_name,
            "history": history,
            "contenders": experiment.contenders,
            "scorer": experiment.scorer,
            "selector": experiment.selector,
        })

    print("All experiments completed.")

    data = {
        "config": config,
        "results": results
    }

    with open("results.pkl", "wb") as file:
        pickle.dump(data, file)

    print("Results saved to results.pkl")