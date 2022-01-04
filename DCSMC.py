import numpy as np
from scipy.stats import expon
from collections import defaultdict
from utils import *
from Particles import * 


class DC_SMC : 
    """
    Python implementation of the Divide & Conquer Sequential Monte Carlo algorithm, based on the java implementation
    of the authors (https://arxiv.org/abs/1406.4993).
    
    Initialization
    -----------
    - tree, TreeDataset, the tree that corresponds to the data to be processed
    - n_particles, int, number of particles to use in the system
    - max_posterior_level: int, Posterior distribution will be drawn for every node with a level inferior to 
        this number.
    - exp_prior_lambda: int, the lambda parameter of the prior over sigma (Exponential distribution).
    """
    def __init__(self, tree, n_particles, exp_prior_lambda = 1, ess_threshold = "default", max_posterior_level = 0) :
        self.tree = tree
        self.n_particles = n_particles
        self.exp_prior_lambda = exp_prior_lambda
        if ess_threshold == 'default' : 
            self.ess_threshold = n_particles/2
        else : 
            assert isinstance(ess_threshold,int),'n_particles should be an integer'
            self.ess_threshold = ess_threshold
        self.posterior = defaultdict(dict)
        for i in range(max_posterior_level+1) : 
            self.posterior[i] = {}
        self.max_posterior_level = max_posterior_level
        self.ess = []

    def draw_sample(self) :
        """
        Recursion of the DC-SMC algorithm presented in the paper over all the nodes of the tree.
        Note that even the function starts from the root, the algorithm will start sampling when reching a leaf node.
        """
        result = self.recurse(self.tree.get_root()[0])
        print(f'LogZ hat: {result.log_Z_hat}')
        return result
    
    def recurse(self, node):
        """
        Main recursion method. Given a leaf node, draw a sample from the LeafParticleApprox class. Given an internal
        node and its childrens, for each particle, draw a variance sample from an exponential(1), then iterates 
        through each children's SMC sampler to calculate the total childs descendant log-likelihood and variance 
        (sum over childrens' descendant) then combine childs messages and create new particle system.
        The recursion stops when it reached the root node.
        
        Parameters
        ----------
        - node: str, the name of the node to process
        
        Returns
        -------
        A StandardParticleApprox object with updated attributes.
        """
        result = StandardParticleApprox(self.n_particles)
        childrens = self.tree.get_child(node)
        logZ = 0
        max_loglike = -np.inf
        if len(childrens) == 0 :
            result = LeafParticleApprox(self.n_particles, self.tree.T.nodes[node]).sample()
        else : 
            samples = []
            for child in childrens : 
                samples.append(self.recurse(child))
                
            result = StandardParticleApprox(self.n_particles)
            
            for sample in samples : 
                logZ+= sample.log_Z_hat
                #print(sample.log_Z_hat)
            for idx in range(self.n_particles) : 
                
                var = self.sample_variance()
                sample_calculators, children_nodes = [], []
                desc_logL = 0
                desc_var = self.variance_log_prior(var)
                for sample in samples : 
                    #Iterate through childs to get their descendant LL and VAR
                    child_particle = sample.particles[idx]
                    sample_calculators.append(child_particle.message) #Get the message from childs
                    children_nodes.append(child_particle.node_info)
                    desc_logL += child_particle.descendant_loglikelihood
                    desc_var += child_particle.descendant_variance
                    
                #Combine the childs messages 
                combined = NormalMessagePassing.combine(sample_calculators.copy(), var)
                combined_logL = combined.loglikelihood
                
                log_weight = combined.loglikelihood

                for child_calculator in sample_calculators : 
                    log_weight = log_weight - child_calculator.loglikelihood
                
                #Create new particle based on childs' descendant LL and VAR, new sampled variance
                #combined message and old messages.
                new_particle = Particle(node, combined, desc_logL, variance = var,
                                       children_nodes = children_nodes, children_messages = sample_calculators,
                                       descendant_variance = desc_var)
                result.particles[idx] = new_particle
                result.probabilities[idx] = log_weight
                
                if (combined_logL + new_particle.descendant_loglikelihood  > max_loglike) : 
                    max_loglike = combined_logL + new_particle.descendant_loglikelihood
                    
        #log weights passed through Softmax to get probabilities
        result.probabilities, log_norm = expnormalize_and_sum(result.probabilities)

        logZ += log_norm - np.log(self.n_particles)
        
        #Calcualtes ess
        ess = self.get_ess(result.probabilities)
        relative_ess = ess / self.n_particles
        if 1-relative_ess > 0.001 : 
            self.ess.append(ess)

        #Resample only if inferior to ess threshold
        if ess < self.ess_threshold : 
            result = self.resample(result)
        
        #Gather node posteriors  
        if self.tree.T.nodes[node]['level'] <= self.max_posterior_level : 
            theta_stats = self.get_theta_stat(node, result.particles)
            sigma_stats = self.get_delta_stat(node, result.particles)
            self.posterior[self.tree.T.nodes[node]['level']][node] = {'theta' : theta_stats, 'delta' : sigma_stats}
        
        #Print iteration
        print(f'NODE : {node}, ESS : {ess}, RESS : {relative_ess}' )
        
        
        if node == 'root'  : 
            self.max_LL = max_loglike
            self.log_Z_hat = logZ
            print(f'MaxLL : {max_loglike}')
            
        result.log_Z_hat = logZ
        
        return result
        
    def get_ess(self, weights) : 
        """Calculates Effective Sample Size"""
        return 1/(np.sum(np.square(weights)))
    
    def sample_variance(self, lambda_=1) : 
        """Draw a sample from an Exponential(lambda_)"""
        return np.random.exponential(scale = 1/lambda_)

    def variance_log_prior(self, var, lambda_ = 1) : 
        """Calculates the log density of an exponential(lambda_) at point var"""
        return expon.logpdf(var, scale = 1/lambda_)
    
    def resample(self, particle_approx) : 
        """Multinomial resampling"""
        resampled_res = StandardParticleApprox(self.n_particles)
        resampled_index = np.random.choice([i for i in range(self.n_particles)], size = self.n_particles)
        
        for idx in range(self.n_particles) : 
            resampled_res.particles[idx] = particle_approx.particles[resampled_index[idx]]
            resampled_res.probabilities[idx] = 1/self.n_particles
            
        return resampled_res
    
    
    def get_theta_stat(self, node, particles) : 
        """Get statistics of theta posterior distriubtion at a given node"""
        children = self.tree.get_child(node)
        mean_samples, samples, variance_samples = [], [], []
        for idx in range(self.n_particles) : 
            particle = particles[idx]
            point = particle.sample()
            mean_point = inverse_transform(point) 
            samples.append(point)
            mean_samples.append(mean_point)
            if len(children) != 0 :
                variance_samples.append(particle.variance)
        return {'mean' : mean_samples, 'samples' : samples, 'variance' : variance_samples}
    
    def get_delta_stat(self, node, particles) : 
        """Get statistics of delta posterior distriubtion at a given node""" 
        children = self.tree.get_child(node)
        n_childrens = len(children)
        delta_samples = np.zeros((n_childrens,self.n_particles))
        for i in range(self.n_particles) : 
            combined_sample = self.sample_children_jointly(particles[i])
            transformed_root = inverse_transform(combined_sample[-1])
            
            for c in range(n_childrens) : 
                transformed_child = inverse_transform(combined_sample[c])
                delta = transformed_child - transformed_root
                delta_samples[c][i] = delta

            
        return {'mean' : delta_samples.mean(), 'std' : np.std(delta_samples), 'samples' : delta_samples.mean(axis = 0)}
        
        
    def sample_children_jointly(self,particle) :
        """Draw a sample of the childs posterior normal distribution at a given particle"""
        root_sample = particle.sample()
        size = len(particle.children_messages)
        result = []
        for c_idx in range(size) : 
            updated_message = particle.message.combine(particle.message, particle.children_messages[c_idx],
                                                       particle.variance, 0.0)
            result.append(updated_message.message[0] + np.random.randn()*np.sqrt(updated_message.message_variance))
        result.append(root_sample)
        return result