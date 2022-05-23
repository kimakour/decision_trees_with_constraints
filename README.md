# decision_trees_with_constraints

A package for decision trees with constraints.

The reasons behind designing this model are the following: 

- Binary decision trees on sklearn need preprocessing for categorical features ( people generally use one hot encoding , mean encoding or ordinal encoding in case of an ordinal meaning of the categorical variable) , this model on the contrary uses the optimal binning method to gather some category levels ( No need for preprocessing, except in cases based on business knowledge) in order to maximize the divergence measure in the objective function. Information Value or Jeffrey’s divergence is set to be the default objective but it can be changed to Jensen Shannon, Hellinger divergence or triangular discrimination. Refer to http://gnpalencia.org/optbinning/ for the optimal binning informations.
- In a business context, we do have constraints in some cases ( like interpretability constraints ). In sklearn, the decision trees use all the features and rank them with their splitting power regardless of the feature meaning, whereas in this model we can add constraints in the tree construction in order to meet our business constraints. For example if we want a feature "A" to appear in the left child 2, we can force the model to learn our modeling function with this "obligation". 


* The aim of this model is thus to bring more control to learn the trees and to meet business expectations in terms of interpretability and constraints. 
