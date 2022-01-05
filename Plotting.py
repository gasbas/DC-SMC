import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

def construct_output_df(posterior_data, target_posterior = 'theta') : 
    
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    
    obs = 0
    idx2 = 0
    for level, data in posterior_data.items() : 
        for node_name, values in data.items() : 

            curr_data = values[target_posterior]
            theta_mean = np.mean(curr_data['samples'])
            theta_std = np.std(curr_data['samples'])
            mean = np.mean(curr_data['mean'])
            mean_std = np.std(curr_data['mean'])
            var = np.mean(curr_data['variance'])
            var_std = np.std(curr_data['variance'])

            if target_posterior == "theta" : 
                curr = pd.DataFrame({
                    'level' : level,
                    'node' : node_name,
                    'theta_mean' : theta_mean,
                    'theta_std' : theta_std,
                    'logistic_theta_mean' : mean,
                    'logistic_theta_std' : mean_std,
                    'sigma_mean' : var,
                    'sigma_std' : var_std
                }, index = [obs])
            else : 
                curr = pd.DataFrame({
                    'level' : level,
                    'node' : node_name,
                    'delta_mean' : mean,
                    'delta_std' : var,
                }, index = [obs])
            obs+=1
            
            #samples
            df1 = df1.append(curr)
            n_samples = len(curr_data['samples'])
            if target_posterior == 'theta' : 
                curr = pd.DataFrame({
                    'level' : [level] * n_samples,
                    'node' : [node_name] * n_samples,
                    'theta' : curr_data['samples'],
                    'logistic_theta' : curr_data['mean'],
                    'sigma' : curr_data['variance']},
                    index = [i for i in range(idx2, idx2 + n_samples)])
            else : 
                curr = pd.DataFrame({
                    'level' : [level] * n_samples,
                    'node' : [node_name] * n_samples,
                    'delta' : curr_data['samples']},
                    index = [i for i in range(idx2, idx2 + n_samples)])
                
            idx2 += n_samples
            df2 = df2.append(curr)
                
    return df1, df2


def density_plot(results, save_path = None, max_sample = 1000, legend_names = None) : 
    
    
    
    if not isinstance(results, list) :  
        tmp = [results]
    else : 
        tmp = results
        
    samples = []
    for idx in range(len(tmp)) : 
        if max_sample > len(tmp[idx].particles) : 
            max_sample = len(tmp[idx].particles)
        samples_LL = []
        for j in range(max_sample) : 
            samples_LL.append(tmp[idx].sample_logdensity())
        samples.append(samples_LL)
        
    colors = ['orange','green','grey','red','blue']


    fig,ax = plt.subplots(figsize = (8,6))
    for idx in range(len(tmp)) : 
        legend_names = [None]*len(results) if legend_names == None else legend_names
        
        ax_s = sns.kdeplot(samples[idx], color = colors[idx], lw = 1.5,
                           label = legend_names[idx], ax =ax, fill = True)
        
    ax.legend(frameon = False)
    ax.set_xlabel('Sampled log density')
    ax.grid(visible = True)
    
    if save_path is not None : 
        plt.savefig(save_path + 'density_plot.png', dpi = 300)
    plt.close()

def trace_plot(result, save_path = None, max_sample = 1000) : 
    
    if max_sample > len(result.particles) : 
        max_sample = len(result.particles)
    samples_LL = []
    for j in range(max_sample) : 
        samples_LL.append(result.sample_logdensity())
        
    plt.Figure(figsize=(8,6))
    plt.plot([i for i in range(len(samples_LL))], samples_LL, c = 'black', lw = 0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Sampled log density')
    plt.grid(visible = True)
    if save_path is not None : 
        plt.savefig(save_path + 'trace_plot.png', dpi = 300)
    plt.close()
def plot_posterior(posterior_data, target_posterior = 'theta', target_func = 'mean', max_cols = 5, 
                   row_names = None, col_names = None, save_path = None) : 
    
    if not isinstance(posterior_data, list) :  
        tmp = [deepcopy(posterior_data)]
    else : 
        tmp = deepcopy(posterior_data)
    n_levels = [len(i.keys()) for i in tmp]

    assert np.unique(n_levels).shape[0] == 1, 'When providing multiple results, make sure that they have similar max posterior level'
    n_cols = 0
    
    for idx in range(len(n_levels)) : 
        n_cols += len(tmp[0][idx])

    if n_cols > max_cols : 
        n_cols = max_cols
    
    fig,ax = plt.subplots(len(tmp), n_cols, figsize = (12,6), sharex=True, sharey=True)
    if not isinstance(posterior_data, list) :  
        ax = [ax]
    if n_cols == 1 : 
        ax = [ax]
    colors = ['orange','green','grey','red','blue']
    
    if target_func == 'variance' : 
        plt.xlim(left = -0.5, right = 5.5) 
        plt.xticks([0,5])
        plt.ylim(top = 3.5)
        plt.yticks([0,1,2,3])
    row_names = [None]*len(tmp) if row_names == None else row_names
    if len(row_names) < len(tmp) : 
        row_names += [None] * len(tmp) - len(row_names)
    col_names = [None]*n_cols if col_names == None else col_names
    if len(col_names) < len(tmp) : 
        col_names += [None] * n_cols - len(col_names)

    ax[-1][0].set_xlabel(list(tmp[idx][0].keys())[0], fontsize = 12)

    for idx in range(len(tmp)) :
        
        ax[idx][0].set_ylabel(row_names[idx], fontsize = 12)
        ax[idx][0].grid(visible = True)
        if idx != len(tmp) -1 : 
            ax[idx][0].tick_params(axis = 'x', width = 0)
        ax[idx][0].tick_params(axis = 'y', width = 1.5)
        
        label_name = "root" if col_names[0] is None else col_names[0]
        ax[idx][0].set_xlabel(label_name, fontsize = 12)
        sns.kdeplot(tmp[idx][0]['root'][target_posterior][target_func], color = colors[idx], lw = 1.5,
                           ax = ax[idx][0], fill = True)
        
        for level in range(0,n_cols-1) : 
            
            node_name = list(tmp[idx][1].keys())[level]
            label_name = node_name if col_names[level+1] is None else col_names[level+1]
            ax[-1][level+1].set_xlabel(label_name, fontsize = 12)
            
            sns.kdeplot(tmp[idx][1][node_name][target_posterior][target_func], color = colors[idx], lw = 1.5,
                               ax =ax[idx][level+1], fill = True)
            ax[idx][level+1].grid(visible = True)
            ax[idx][level+1].tick_params(axis = 'y', width = 0)
            if idx != len(tmp) -1 : 
                ax[idx][level+1].tick_params(axis = 'x', width = 0)
            else : 
                ax[idx][level+1].tick_params(axis = 'x', width = 1.5)

    plt.subplots_adjust(wspace=0.2, hspace=0.05)
    
    if save_path is not None : 
        plt.savefig(save_path + f'{target_posterior}_{target_func}_plot.png', dpi = 300)
    plt.close()
    
def ess_plot(ess, threshold, save_path = None) : 

    plt.Figure(figsize=(8,6))
    plt.plot(ess, c = 'black', lw = 1.2)
    plt.xlabel('Iteration')
    plt.ylabel('Effective Sample Size')
    plt.hlines(xmin = 0, xmax = len(ess), y = threshold, linestyle = '--', color = 'grey', alpha = 1, lw = 1.5)
    plt.grid(visible = True)
    if save_path is not None : 
        plt.savefig(save_path + 'ess_plot.png', dpi = 300)
    plt.close()