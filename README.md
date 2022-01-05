# Divide & Conquer Sequential Monte Carlo

Project for the Hidden Markov Models course at M2 Data-Science, Institut Polytechnique de Paris. We applied Divide & Conquer SMC algorithm to hierarchical data as presented in Section 5.2 of the original paper (https://arxiv.org/abs/1406.4993).

# Installation

This project requires Python 3.9. 
We suggest to create a new python 3.9 virtual environment.

Clone the project using `git clone https://github.com/gasbas/DC-SMC.git`

Install dependancies using `pip install -r requirements.txt`
# Preprocessing

At the root of the folder, run:
``` 
cd preprocess/

./preprocess_DC_data.sh
``` 
This will download 2017, 2018 and 2019 data from the Washington DC Education Office website (https://osse.dc.gov/), as well as a JSON file (hosted on google drive) that contains a few replacements name in order to match school names between different years.

A preprocessed dataset is then created directly in the `preprocess` folder. 

# Running the tests

You can reproduce our examples by running 
``` 
python main.py 
```
This will launch seeded runs and simulation that we used for the report.

It is also possible to run the tests with different parameters. The full call to the main file is: 

```
python main.py --data_path XXX --save_path XXX --n_particles XXX --n_sim XXX
--exp_prior XXX --ess_threshold XXX
```
- *data_path* is the path to the preprocessed data.
-  *save_path* is the path where the results will be stored.
-  *n_particles* is the number of particles to use during the simulations. Default to 10000
-  *n_sim* is the number of simulations to calculate variability over Log Z hat.
-  *exp_prior* is the parameter of the Exponential distribution used as a prior for sigma.
-   *ess_threshold* refers to the threshold used for resampling. Default to n_particles/2


# Authors

COURRIER Violaine, OULHAJ Hamza, MICHEL Gaspard