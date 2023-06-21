# Import general packages
import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Define constants
# Genetic algorithm meta-parameters
K_BEST = 16  # Number of resulting features
M_BEST = 3  # Number of rhe best models
POP_SIZE = 14  # Number of various models min = M_BEST
MUT_PROB = 6 * 1 / K_BEST  # Chance of mutation for each feature WAS:2
MAX_ITER = 10000  # Times POP_SIZE equals number of fits


# Class Genetic feature selection
class GeneticFS:
    def __init__(self, n_features):
        self.k_best = K_BEST
        self.m_models = M_BEST
        self.pop_size = POP_SIZE
        self.mut_prob = MUT_PROB
        self.max_iter = MAX_ITER
        self.plot_cap = True  # Plot progressive epochs
        self.pop_report = True  # Print them
        self.warnings_off = True
        self.__create_population(n_features)
        # There is no need to evaluate the population now
        self.scores = np.zeros(self.pop_size)  # The score is positive, the bigger the better
        # Collect statistics of best features over populations
        self.topM_score = np.zeros(self.m_models)
        self.topM_feat = np.zeros([self.m_models, n_features])

    #  --------------------------------------------------------------
    def fit(self, X, Y, clf):
        n_features = X.shape[1]
        for i in range(MAX_ITER):
            offspring = self.__cross_mutate(n_features)
            scores_offspring = self.fit_evaluate(X, Y, clf, offspring)
            self.__update_population(offspring, scores_offspring)
            self.__update_score_stat()
            # ---
            if self.plot_cap:
                values = self.topM_feat[1, :] / np.max(self.topM_feat[1, :])
                self.plot_biosemi(values)
            if self.pop_report:
                print(f'Iteration: {i}, score of best: {self.topM_score[1]}, features: {self.topM_feat[1, :]}')
                print(f'Scores: {self.scores}')
        # Save and show results
        # save_population() # this function is commented
        nameFeatures = [f"f_{i}" for i in range(1, 65)]  # FIXIT
        self.print_statistics(self.topM_feat[1], nameFeatures)

    #  ----------------------------- ---------------------------------
    def __create_population(self, n_features):
        # Create a random population to start with
        population = np.empty([self.pop_size, self.k_best], dtype=int)
        for i in range(self.pop_size):
            population[i, :] = np.random.choice(range(n_features), size=self.k_best, replace=False)
        self.population = population

    #  ----------------------------- ---------------------------------
    def __cross_mutate(self, n_features):
        offspring = np.empty_like(self.population)
        # First genetic step is to crossover all members of the population
        for idx, mom in enumerate(self.population):  # For each mom
            dad = self.population[np.random.choice(len(self.population)), :]  # Get a random dad
            # k_best = self.population.shape[1] # Number of allowed features matches with dim
            rnd = np.random.choice(self.k_best)  # Pick some chromosomes
            kid = np.concatenate((mom[:rnd], dad[-(self.k_best - rnd):]))  # Make a kid
            offspring[idx, :] = kid  # Put it to new tribe
            # print(idx, rnd, mom, dad, kid)
        # Second genetic step is to mutate each member of the new population
        for idx, kid in enumerate(offspring):
            mutation = np.random.choice([0, 1], size=self.k_best,
                                        p=[self.mut_prob, 1 - self.mut_prob])  # Set random chromosomes
            unused = np.setdiff1d(np.arange(n_features), kid)  # Find the new chromosomes
            if len(unused) >= np.sum(mutation):  # is there enough new chromosomes
                deviation = np.random.choice(unused, size=np.sum(mutation), replace=False)  # select a few
                # oldkid = np.array(kid) # debug
                _ = np.where(mutation == 1)[0]
                kid[_] = deviation
                # print(mutation, kid, deviation) # debug
            else:
                print('Mutation rate is too high, no new population member generated')  # Let the kid be same
            offspring[idx, :] = kid  # Put non-mutated kid back to the population offspring
        return offspring

    #  ----------------------------- ---------------------------------
    def fit_evaluate(self, X, Y, clf, offspring):
        scores_offspring = np.zeros(len(offspring))
        # Third genetic step is to evaluate the quality of each member
        for idx, kid in enumerate(offspring):
            X_cut = X[:, kid]
            auc = self.__one_cls(X_cut, Y, clf)
            scores_offspring[idx] = auc  # Evaluate each member of the offspring
        return scores_offspring

    #  ----------------------------- ---------------------------------
    # Classify for parameter optimization and accuracy evaluation
    def __one_cls(self, X, Y, clf):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
        if self.warnings_off:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train)
        # Part for accuracy
        # y_pred = clf.predict(X_test)
        # acc = np.mean(y_test == y_pred)
        # print(f'Accuracy = {acc}')
        # Part for AUC
        y_pred = clf.predict_proba(X_test)[::, 1]  # Probability for AUC
        auc = metrics.roc_auc_score(y_test, y_pred)
        # fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        # print(f'Accuracy = {auc}')
        return auc  # acc

    #  ----------------------------- ---------------------------------
    def __update_population(self, offspring, scores_offspring):
        # Sort and keep the best of old population and offspring
        # Join the old tribe and the offspring tribe
        population = np.concatenate((self.population, offspring), axis=0)
        scores = np.concatenate((self.scores, scores_offspring), axis=0)
        idx = np.argsort(-scores)[:len(scores_offspring)]  # The bigger score the better
        self.population = population[idx, :]  # Select the best members
        self.scores = scores[idx]  # Keep the best scores (cut to offspring size)
        self.population = population[idx, :]  # Select the best members

    #  ----------------------------- ---------------------------------
    def __update_score_stat(self):
        # Statistics each round for the best one and the population
        for i, _ in enumerate(self.topM_score):
            self.topM_score[i] = self.scores[i]
            kid = self.population[i, :]  # Just copy the first M scores to save
            self.topM_feat[i, kid] += 1  # Increase frequency of each selected feature

    #  ----------------------------- ---------------------------------
    # Print the names of the most frequent features and their occurrence
    def print_statistics(self, frequencies, labels):  # The last is not used
        zipped = zip(frequencies, labels)  # Each feature has its frequency
        sorted_zipped = sorted(zipped, reverse=True)  # Sort to show the best
        # sorted_frequencies, sorted_labels = zip(*sorted_zipped)
        # Unzip the sorted zipped array
        # Print the sorted frequencies and labels
        # print("Sorted Frequencies:", sorted_frequencies)
        # print("Sorted Labels:", sorted_labels)
        for frequency, label in sorted_zipped:
            print((label, frequency))

    def plot_biosemi(self, vector):
        matrix = np.reshape(vector, (8, 8))  # This reshape will be used for visualization
        plt.figure(1, figsize=(3, 3))
        plt.imshow(matrix, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.show()
#  ----------------------------- ---------------------------------
#  ----------------------------- ---------------------------------
#  ----------------------------- ---------------------------------
#  ----------------------------- ---------------------------------
# def save_population(self):
# # Since genetics take long time, it is important to save the intermediate results
#  df_population = pd.DataFrame(self.population)      # Save the population with the features
#  df_scores     = pd.DataFrame(self.scores)          # Save the scores to start from
#  now = datetime.datetime.now()                 # Get today' time to make the filename
#  date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S") # Format it
#
#      fn_population = f'Population_{date_time_str}.xlsx'
#      fn_scores = f'Scores_{date_time_str}.xlsx'
#      df_population.to_excel(fn_population, index=False)  # Save to an Excel file
#      df_scores.to_excel(fn_scores, index=False)
#      files.download(fn_population)
#      files.download(fn_scores)
#      return
#  ----------------------------- ---------------------------------
# def load_population(self, population):
#  return
