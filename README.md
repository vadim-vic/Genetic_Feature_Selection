# Genetic Feature Selection
a silplest version of the algorithm 

The concept of genetic optimization is to create a population of models and randomly exchange their features with the accuracy evaluation, selecting the best models after every new population offspring. According to Holland's schema theorem, the number of informative features will [exponentially grow in the best models](https://dynamics.org/Altenberg/FILES/LeeSTPT.pdf)
The genetic algorithm selects a limited number of features, a subset from a large feature set (for this slide, it is 16 out of 512). First, a population of models exists: randomly selected subsets of features. Second, for each member of the population, a mom and a random dad were selected. They exchange a random number of features (analog is the chromosome cross-over). A kid carries the same number of features. Some features of each kid in the offspring are randomly replaced with new ones (analog is the mutation). The quality of each kid in the offspring is evaluated (tune parameters using the test data according to the likelihood and evaluate the quality using the test data according to AUC (or another quality criterion); cross-validate for each model). Make the new population of the same size from the best members of the old population and the offspring (analog is fitness). Repeat the second step. The stop criterion is the given number of generated populations (10,000, for example) or convergence in the feature occurrence. 
The properties of the genetic optimization algorithm in comparison to the other feature selection algorithms are comprehensively analyzed in 

1. Katrutsa A.M., Strijov V.V. Comprehensive study of feature selection methods to solve multicollinearity problem according to evaluation criteria // Expert Systems with Applications, 2017, 76: 1-11. [DOI: 10.1016/j.eswa.2017.01.048](https://doi.org/10.1016/j.eswa.2017.01.048)

The reason when the genetic feature selection algorithm is the high number of correlated features. Also, when the number of features is comparable to the number of objects in the dataset. The Elastic Net algorithm, used for the feature selection, delivers unstable results: the regularization path significantly varies over slight changes in data (and over cross-validation). Therefore, there are only two solutions for structure optimization 1) a discrete genetic feature selection algorithm and 2) a quadratic programming feature selection algorithm. The other algorithms are tested here:

2. Katrutsa A.M., Strijov V.V. Stresstest procedure for feature selection algorithms // Chemometrics and Intelligent Laboratory Systems, 2015, 142: 172-183. [DOI: 10.1016/j.chemolab.2015.01.018](https://doi.org/10.1016/j.chemolab.2015.01.018)

An example of the genetic model generation for information retrieval is here

3. Kulunchakov A.S., Strijov V.V. Generation of simple structured Information Retrieval functions by genetic algorithm without stagnation // Expert Systems with Applications, 2017, 85: 221-230. (DOI: 10.1016/j.eswa.2017.05.019)[https://doi.org/10.1016/j.eswa.2017.05.019]

