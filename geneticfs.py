# Import general packages
import numpy as np
# Define constants
# Genetic algorithm metaparameters
K_BEST =  16     # Number of resulting features
M_BEST = 3      # Number of rhe best models
POP_SIZE = 14   # Number of variuos models min = M_BEST
MUT_PROB = 6*1/K_BEST  # Chance of mutation for each feature WAS:2
MAX_ITER = 10000   # Times POP_SIZE equals number of fits
# not used NEXT_ITER = 10.  # Run limites number populations to try

# Class Genetic feature selection
class GeneticFS:
  #  ----------------------------- ---------------------------------
  def __init__(self, n_Features):
        self.k_features = K_BEST
        self.m_models   = M_BEST
        self.pop_size   = POP_SIZE
        self.mut_prob   = MUT_PROB
        self.max_iter   = MAX_ITER
        self.plot_cap   = True          # Print
        self.pop_report = True
        self.population, self.scores  = create_population(n_Features)
        self.topM_score = np.zeros(self.m_models)
        self.topM_feat  = np.zeros([self.m_models,n_Features])
  #  ----------------------------- ---------------------------------
  def fit(self, X, Y, clf):
    for i in range(MAX_ITER):
      offspring           = cross_mutate(population, n_Features, MUT_PROB)
      scores_offspring    = fit_evaluate(X,Y, clf, offspring)
      __update_population(offspring, scores_offspring)
      __update_score_stat(population, scores, topM_score, topM_feat)
      # ---
      if self.plot_cap:
        values = topM_feat[1,:] / np.max(topM_feat[1,:])
        plot_biosemi(values)
      if self.pop_report:
        print(f'Iteration: {i}, score of best: {topM_score[1]}, features: {topM_feat[1,:]}')
        print(f'Scores: {scores}')
    # Save and show results
    save_population()
    print_statistics(self.topM_feat[1], nameFeatures)
  #  ----------------------------- ---------------------------------
  def __create_population(self, n_Features):
    # Create a random population to start with
    population = np.empty([self.pop_size, self.k_best], dtype=int)
    for i in range(self.pop_size):
      population[i, :] = np.random.choice(range(n_Features), size=self.k_best, replace=False)
    scores =  np.zeros(self.pop_size) # There is no need to evaluate now (score positive, the bigger the better)
    return(population, scores)
  #  ----------------------------- ---------------------------------
  def __cross_mutate(self, population, n_Features, MUT_PROB):
    offspring = np.empty_like(self.population)
    # First genetic step is to crossover all members of the population
    for idx, mom  in enumerate(self.population):           # For each mom
      dad = self.population[np.random.choice(len(self.population)), :] # Get a random dad
      kBest = self.population.shape[1]               # Number of allowed features
      rnd = np.random.choice(kBest)                  # Pick some chromosomes
      kid = np.concatenate((mom[:rnd], dad[-(kBest-rnd):])) # Make a kid
      offspring[idx, :] = kid                         # Put it to new tribe
      #print(idx, rnd, mom, dad, kid)
    # Second genetic step is to mutate each member of the new population
    for idx, kid  in enumerate(offspring):
      mutation = np.random.choice([0, 1], size=kBest, p=[self.mut_prob, 1-self.mut_prob]) # Set random chromosomes
      unused = np.setdiff1d(np.arange(n_Features), kid)  # Find the new chromosomes
      if len(unused) >= np.sum(mutation):                # is there enough new chromosomes
        deviation = np.random.choice(unused, size=np.sum(mutation), replace=False)  # select a few
        #oldkid = np.array(kid) # debug
        _ = np.where(mutation == 1)[0]
        kid[_] = deviation
        #print(mutation, oldkid,  kid, deviation) # debug
      else:
        print('Mutation rate is too high, no new population member generated') # Let the kid be same
      offspring[idx, :] = kid # Put mutated kid back to the population offspring
    return offspring
  #  ----------------------------- ---------------------------------
  def fit_evaluate(self, X, Y, clf, offspring):
    scores_offspring = np.zeros(len(offspring))
    # Third genetic step is to evaluate the quality of each member
    for idx, kid in enumerate(offspring):
      X_cut = X[:, kid]
      acc = __one_cls(X_cut, Y, clf)
      scores_offspring[idx] = acc # Evaluate each member of the offspring
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
    #preds = clf.predict_proba(X)[::,1] # Probabilty for AUC
    preds = clf.predict(X_test)
    acc = np.mean(preds == y_test)
    #print(f'Accur = {acc}')
    return acc
  #  ----------------------------- ---------------------------------
  def __update_population(self, offspring, scores_offspring):
    # Sort and keep the best of old population and offspring
    # Join the old tribe and the offspring tribe
    population = np.concatenate((self.population, offspring), axis=0)
    scores     = np.concatenate((self.scores, scores_offspring), axis = 0)
    idx = np.argsort(-scores)[:len(scores_offspring)] # The bigger score the better
    self.scores     = scores[idx]  # Keep the best scores (cut to offspring size)
    self.population = population[idx,:] # Select the best members
    # return population, scores
  #  ----------------------------- ---------------------------------
  def __update_score_stat(self):
    # Statistics each round for the best one and the population
    for i, _ in enumerate(topM_score):
      topM_score[i] = self.scores[i]
      kid = self.population[i, :]  # Just copy the first M scores to save
      # kid = np.array([3,4,5],  dtype=int) # Just for test
      topM_feat[i, kid] += 1  # Increase frequency of each selected feature
    return topM_score, topM_feat
  #  ----------------------------- ---------------------------------
  def save_population(self):
    # Since genetics take long time, it is important to save the intermediate results
    df_population = pd.DataFrame(self.population)      # Save the population with the features
    df_scores     = pd.DataFrame(self.scores)          # Save the scores to start from
    now = datetime.datetime.now()                 # Get today' time to make the filename
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S") # Format it

    fn_population = f'Population_{date_time_str}.xlsx'
    fn_scores     = f'Scores_{date_time_str}.xlsx'
    df_population.to_excel(fn_population, index=False) # Save to an Excel file
    df_scores.to_excel(fn_scores, index=False)
    files.download(fn_population)
    files.download(fn_scores)
    return
  #  ----------------------------- ---------------------------------
  #def load_population(self, population):
  #  return
  #  ----------------------------- ---------------------------------
  def print_statistics(self, frequencies, labels): # The last is not used
    # Print the names of the most frequest featues and their occurance
    # Thanks of ChatGPT
    zipped = zip(frequencies, labels) # Each feature has its frequency
    sorted_zipped = sorted(zipped, reverse=True) # Sort to show the best
    # Unzip the sorted zipped array
    sorted_frequencies, sorted_labels = zip(*sorted_zipped)
    # Print the sorted frequencies and labels
    # print("Sorted Frequencies:", sorted_frequencies)
    # print("Sorted Labels:", sorted_labels)
    for frequency, label in sorted_zipped:
          print((label, frequency))
    return
  #  ----------------------------- ---------------------------------
  #  ----------------------------- ---------------------------------
  #  ----------------------------- ---------------------------------