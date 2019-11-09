'''
DESCRIPTION
-----------
This script displays how to use the auto_design machine learning package for automatically
building a neural netwrok pipeline (scaler + model + feature reduction stage) and tuning it.
'''


#%% code imports
import numpy as np 
from auto_design.utils import design
from sklearn.datasets import load_boston


#%% main code section
if(__name__=='__main__'):
    # create random data for testing
    x, y = load_boston(return_X_y=True)
    y = y.reshape(-1, 1)
    # specify problem type
    problem_type = 'regression'
    # specify the size of population and number of generations
    size_population = 15
    number_generations = 10
    # design ML pipeline using auto_design module
    ml_design, log = design(x,
                            y,
                            problem_type, 
                            size_population, 
                            number_generations)