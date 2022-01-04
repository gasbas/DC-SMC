import numpy as np
import pandas as pd
import os
import argparse 


from Dataset import TreeDataset
from utils import * 
from DCSMC import DC_SMC
from Plotting import * 

parser = argparse.ArgumentParser(description='DC-SMC implementation for hierarchical model on Washington DC data')

# Data related args
parser.add_argument('--data_path', type = str, default = 'preprocess/preprocessed_DC_data.csv', help = 'path to preprocessed data')
parser.add_argument('--save_path', type = str, default = 'results/', help = 'path where the results will be stored')

# Model related args

parser.add_argument('--n_particles', type = int, default = 10000, help = 'number of particles to use in simulations')
parser.add_argument('--n_sim', type = int, default = 50, help = 'number of simulations to run to get log Z hat variability')
parser.add_argument('--exp_prior', type = float, default = 1.0, help = 'parameter of the exponential distribution used as prior of sigma')
parser.add_argument('--ess_threshold', default = "default", help = 'if effective sample size is below this threshold, use multinomial resample. Default to n_particles/2')

args = parser.parse_args()

if __name__ == '__main__' : 

    # Create result folders
    if not os.path.exists(os.path.join(args.save_path,'figures')) : 
        os.makedirs(os.path.join(args.save_path,'figures'))
    if not os.path.exists(os.path.join(args.save_path,'tables')) : 
        os.makedirs(os.path.join(args.save_path,'tables'))

    # Load dataset and create tree
    tree = TreeDataset(args.data_path)

    #### Step 1. results for varying number of particles
    print('STEP 1. MULTIPLE NUMBER OF PARTICLES')

    all_posteriors = []
    all_results = []
    print('\n\nFITTING MODEL WITH N_PARTICLES = 100')

    model = DC_SMC(tree, n_particles = 100, exp_prior_lambda = args.exp_prior, ess_threshold=args.ess_threshold, seed = 1, max_posterior_level = 1)
    all_results.append(model.draw_sample())
    all_posteriors.append(model.posterior.copy())
    del model
    print('\n\nFITTING MODEL WITH N_PARTICLES = 1000')

    model = DC_SMC(tree, n_particles = 1000, exp_prior_lambda = args.exp_prior, ess_threshold=args.ess_threshold, seed = 1, max_posterior_level = 1)
    all_results.append(model.draw_sample())
    all_posteriors.append(model.posterior.copy())
    del model
    print('\n\nFITTING MODEL WITH N_PARTICLES = 10000')

    model = DC_SMC(tree, n_particles = 10000, exp_prior_lambda = args.exp_prior, ess_threshold=args.ess_threshold, seed = 1, max_posterior_level = 1)
    result = model.draw_sample()
    all_results.append(result)
    all_posteriors.append(model.posterior)

    ### Plotting results
    print('\n\nPLOTTING POSTERIOR DISTRIBUTIONS')

    plot_posterior(all_posteriors, target_posterior='theta', target_func = 'mean', row_names = ['N = 100', 'N = 1000', 'N = 10000'],
                   col_names = ['DC', 'Ward 1', 'Ward 2', 'Ward 3', 'Ward 4', 'Ward 5', 'Ward 6', 'Ward 7', 'Ward 8'], max_cols = 100,
                   save_path = os.path.join(args.save_path,'figures/'))

    
    plot_posterior(all_posteriors, target_posterior='theta', target_func = 'variance', row_names = ['N = 100', 'N = 1000', 'N = 10000'],
                   col_names = ['DC', 'Ward 1', 'Ward 2', 'Ward 3', 'Ward 4', 'Ward 5', 'Ward 6', 'Ward 7', 'Ward 8'], max_cols=100,
                   save_path = os.path.join(args.save_path,'figures/'))

    print('\n\nPLOTTING ADDITIONAL PLOTS WITH N = 10000')

    threshold = 10000 / 2 if args.ess_threshold == 'default' else args.threshold
    ess_plot(model.ess, save_path = os.path.join(args.save_path,'figures/'), threshold = threshold)
    density_plot(all_results, save_path = os.path.join(args.save_path,'figures/'), legend_names= ['N = 100', 'N = 1000', 'N = 10000'])
    trace_plot(result, save_path = os.path.join(args.save_path,'figures/'))

    ### Construct tables and save them
    
    theta_desc, theta_samples = construct_output_df(model.posterior, target_posterior= 'theta')
    theta_desc.to_csv(os.path.join(args.save_path,'tables','theta_stat_desc.csv'),index = False)
    theta_samples.to_csv(os.path.join(args.save_path,'tables','theta_samples.csv'),index = False)

    delta_desc, delta_samples = construct_output_df(model.posterior, target_posterior= 'delta')
    delta_desc.to_csv(os.path.join(args.save_path,'tables','delta_stat_desc.csv'),index = False)
    delta_samples.to_csv(os.path.join(args.save_path,'tables','delta_samples.csv'),index = False)

    del model,result, all_results, all_posteriors

    ### Step 2. Simulating to get uncertainity estimate on Log Z hat

    print(f'\n\nSTARTING SIMULATING FOR {args.n_sim} ITERATION')
    all_log_Z = np.zeros(args.n_sim)

    for iteration in range(args.n_sim) : 
        print('\n\n-------------------------- ITERATION {iteration} -------------------------\n\n')
        model = DC_SMC(tree, args.n_particles, exp_prior_lambda = args.exp_prior, ess_threshold=args.ess_threshold, seed = iteration)
        result = model.draw_sample()
        all_log_Z[iteration] = result.log_Z_hat

    df = pd.DataFrame({'log_Z_mean' : all_log_Z.mean(), 'log_Z_std' : all_log_Z.std()}, index = [0])
    df.to_csv(os.path.join(args.save_path,'tables','logZhat_desc.csv'),index = False)

    df = pd.DataFrame({'log_Z_sim' : all_log_Z}, index = [i for i in range(args.n_sim)])
    df.to_csv(os.path.join(args.save_path,'tables','logZhat_samples.csv'),index = False)

