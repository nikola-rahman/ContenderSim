import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# class for selecting contender based on score
# it takes in scores and returns the index of the selected contender
# it has several strategies for selecting the contender: greedy and probabilistic (with temperature and top_p)
class ContenderSelector:
    def __init__(self, strategy, contenders, scorer, **kwargs):
        self.strategy = strategy
        self.contenders = contenders
        self.scorer = scorer
        self.kwargs = kwargs

    # select the contender based on the strategy
    def select(self):
        scores = np.array([self.scorer.score_func(c) for c in self.contenders])
        # work_done = np.array([c.total_responses_made for c in contenders])
        # work_done_normalized = work_done / (np.sum(work_done) + 1)

        if self.strategy == "greedy":
            return self.select_greedy(scores)
        elif self.strategy == "probabilistic":
            temp = self.kwargs.get("temperature", 1.0)
            top_p = self.kwargs.get("top_p", 1.0)
            probs = self.score_to_prob(scores, temp, top_p)

            return self.select_probabilistic(probs)
        else:
            raise ValueError(f"Invalid strategy '{self.strategy}'")

    # select the contender based on the greedy strategy
    def select_greedy(self, scores):
        return np.argmax(scores)
    
    def score_to_prob(self, scores, temperature, top_p):
        scores = scores - np.max(scores)
        probs = np.exp(scores / (temperature+1e-6)) # to make it deterministic set temperature to 0
        probs = probs / np.sum(probs)

        if top_p < 1.0:
            probs = self.top_p_filter(probs, top_p)

        return probs
    
    # filter out the low probability contenders
    def top_p_filter(self, probs, top_p):
        sorted_probs = np.sort(probs)[::-1]
        cumsum = np.cumsum(sorted_probs)
        threshold = np.min(np.where(cumsum >= top_p))
        threshold = max(1, threshold)
        probs[probs < sorted_probs[threshold]] = 0
        probs = probs / np.sum(probs)
        return probs

    # select the contender based on the probabilistic strategy
    def select_probabilistic(self, probs):
        # if all scores are zero, return a random contender
        if np.sum(probs) == 0:
            return np.random.choice(len(probs))

        return np.random.choice(len(probs), p=probs)

def min_score_to_top_p(scores, min_score_threshold, temperature=1.0):
    # Adjust scores for numerical stability
    scores_adj = scores - np.max(scores)
    # Compute unnormalized probabilities
    probs_unnorm = np.exp(scores_adj / temperature)
    # Normalize probabilities
    probs = probs_unnorm / np.sum(probs_unnorm)
    
    # Sort probabilities in decreasing order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Map original indices to positions in sorted_probs
    positions = np.empty_like(sorted_indices)
    positions[sorted_indices] = np.arange(len(sorted_indices))
    
    # Identify indices of scores above the threshold
    desired_indices = np.where(scores >= min_score_threshold)[0]
    # Get positions of these indices in sorted_probs
    desired_positions = positions[desired_indices]
    # Find the highest position among them
    idx_desired = np.max(desired_positions)
    
    # Compute cumulative sum of sorted probabilities
    cumsum = np.cumsum(sorted_probs)
    # The minimal top_p is the cumulative probability up to idx_desired
    top_p = cumsum[idx_desired]
    
    return top_p


# example usage
if __name__ == '__main__':
    # generate random scores with normal distribution mean 1 and std 0.5
    n_contenders = 200
    scores = np.random.normal(size=n_contenders, loc=0, scale=0.3)
    scores = np.clip(scores, 0, 1)
    scores **= 0.1 # this will make the scores more skewed towards 1
    scores *= 2
    print(np.max(scores), np.min(scores))
    # scores = np.array([1, 2])
    # print("Scores:", scores)
    # probs
    temp = 1.0
    # top_p = 0.2
    # insead of hardcoding top_p, we can calculate it based score threshold
    min_score = 1.8
    top_p = min_score_to_top_p(scores, min_score, temperature=temp)
    print(f"Top P: {top_p:.8f} for min score: {min_score}.")

    # what should top_p be so that all contenders with score less than min_score are filtered out?

    probs = ContenderSelector("probabilistic").score_to_prob(scores, temperature=temp, top_p=top_p)
    print(100*"-")
    for score, prob in sorted(zip(scores, probs), key=lambda x: x[0]):
        print(f"Score: {score:.3f} -> Prob: {prob:.8f}")
    print(100*"-")
    
    # create a selector with greedy strategy
    greedy_selector = ContenderSelector("greedy")
    probablistic_selector = ContenderSelector("probabilistic", temperature=temp, top_p=top_p)

    n_experiments = 1000
    greedy_selections = []
    probablistic_selections = []

    for _ in range(n_experiments):
        greedy_selections.append(greedy_selector.select(scores))
        probablistic_selections.append(probablistic_selector.select(scores))

    # plot distributions of scores, and selections
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(scores, bins=20, color='skyblue', alpha=0.7)
    plt.title("Scores Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 2)
    # plot the selections normalized to the number of experiments
    plt.hist(greedy_selections, bins=10, color='salmon', alpha=0.7) #, density=True)
    plt.title("Greedy Selections")
    plt.xlabel("Contender Index")
    plt.ylabel("Frequency")

    plt.subplot(1, 3, 3)
    plt.hist(probablistic_selections, bins=10, color='green', alpha=0.7) #, density=True)
    plt.title("Probabilistic Selections")
    plt.xlabel("Contender Index")
    plt.ylabel("Frequency")

    plt.tight_layout()
    # plt.show()

    # group by score and plot the count ofselections
    df = pd.DataFrame({"contender_id": range(n_contenders), "score": scores})
    df_selections = pd.DataFrame({
        "greedy": greedy_selections,
        "probabilistic": probablistic_selections
    })

    df['score_bin'] = pd.cut(df['score'], bins=10, labels=[f"Bin {i}" for i in range(10)])

    # now we need to join the selections with the scores so that for each bin we have the number of counts it was selected like this
    # bin 1: 5 (selections)
    # bin 2: 10
    # etc

    df_selections = df_selections.melt(var_name="strategy", value_name="contender_id")
    df_selections = df_selections.merge(df, left_on="contender_id", right_on="contender_id")

    print(df.head())

    print(df['score_bin'].value_counts())

    print(df_selections.head())

    # plot the selections for each bin
    plt.figure(figsize=(12, 6))
    for i, (name, group) in enumerate(df_selections.groupby("strategy")):
        plt.subplot(1, 2, i+1)
        group['score_bin'].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title(f"{name} Selections by Score Bin")
        plt.xlabel("Score Bin")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)

    plt.tight_layout()  
    # plt.show()

    print(df_selections.tail())
    # plot scores on x axis and count of selections on y axis
    df_score_counts = df_selections.groupby(["score", "strategy"]).size().reset_index(name="count")
    df_score_counts['frequency'] = df_score_counts['count'] / n_experiments
    plt.figure(figsize=(10, 6))
    # scatter plot of scores and selections from df_score_counts
    sns.scatterplot(x='score', y='frequency', hue='strategy', data=df_score_counts, palette="viridis")
    # plot expected frequencies based on probs
    df_expected = pd.DataFrame({
        "score": scores,
        "frequency": probs
    })
    sns.scatterplot(x='score', y='frequency', data=df_expected, color='red', label='Expected Frequency')
    plt.title("Contender Selections by Score")
    plt.xlabel("Score")

    plt.show()