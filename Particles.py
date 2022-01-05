import numpy as np
import scipy
from scipy.special import comb, log_softmax
from utils import transform

class Particle : 
    """
    A Particle class that contains sampled informations.
    Attributes
    ----------
    - node_info: dictionary, contains the number of successes, trials and the level of the current node
    - message: NormalMessagePassing, the current sample message
    - variance: float, variance of the sample
    - children_messages: list<NormalMessagePassing>, contains the childrens messages
    - children_nodes: list, node name of the childrens
    - descendant_loglikelihood; float, sum over childs loglikelihood
    - descendant_variance: flaot, sum over childs variance
    """
    def __init__(self, node_info, message, descendant_loglikelihood, children_messages = None, variance = None, 
                 children_nodes = None, descendant_variance = None) : 
        self.node_info = node_info 
        self.message = message
        self.variance = variance
        self.children_messages = children_messages
        self.children_nodes = children_nodes
        self.descendant_loglikelihood = descendant_loglikelihood
        self.descendant_variance = descendant_variance
        
    def log_density(self) : 
        return self.descendant_loglikelihood + self.message.loglikelihood + self.descendant_variance
    
    def sample(self) : 
        return self.message.message[0] + np.random.normal()*np.sqrt(self.message.message_variance)

class NormalMessagePassing : 
    """
    Message Passing algorithm for Normal distribution.
    
    Attributes
    ----------
    - message: list, a list containing the message value (float)
    - message_variance: float, variance of the message
    - loglikelihood: float, the message loglikelihood
    """
    def __init__(self, message, message_variance, loglikelihood) : 
        self.message = message
        self.message_variance = message_variance
        self.loglikelihood = loglikelihood
        
    @staticmethod
    def calculate( leaf1, leaf2, v1, v2) : 
        """
        Combines the message of two messages into a new one.
        Parameters
        ----------
        - leaf1, leaf2: NormalMessagePassing
        - v1, v2: variance parameter of the messages
        Returns
        ----------
        combined NormalMessagePassing
        """
        #print(leaf1.message)
        #print(leaf2.message)
        logl = 0 
        var1 = leaf1.message_variance
        var2 = leaf2.message_variance
        var = 1/(var1+v1) + 1/(var2+v2)
        new_message_variance = 1/var
        
        mean1 = leaf1.message[0]
        mean2 = leaf2.message[0]
        
        message = (mean1 / (var1+v1) + mean2 / (var2+v2) ) /var 
        
        logl = NormalMessagePassing.log_normal_density(mean1 - mean2,0, (v1+var1 + v2+ var2))
        
        logl += leaf1.loglikelihood + leaf2.loglikelihood
        
        return NormalMessagePassing([message], new_message_variance, logl)
    
    @staticmethod
    def log_normal_density(x, mean, var) :
        """Normal log density at point x"""
        return -0.5*(x-mean)*(x-mean)/var -0.5*np.log(2*np.pi * var)
    
    @staticmethod
    def combine(children, variance, children_2 = None, var_2 = None) : 
        """Combine multiple messages (not limited to 2 as in calculates method) via recursion"""
        if all([children_2 != None, var_2 != None]) : 
            return NormalMessagePassing.calculate(children, variance, children_2, var_2)
        
        if len(children) == 1 :
            current = children[0]
            return NormalMessagePassing(current.message, current.message_variance + variance, 
                                       current.loglikelihood)
        

        
        first = children.pop(0)
        second = children.pop(0)
        current = NormalMessagePassing.combine(first,second, variance, variance)
        
        while len(children) != 0 : 
            next = children.pop(0)
            current = NormalMessagePassing.combine(current, next, 0.0, variance)
        return current

                
class StandardParticleApprox : 
    """
    SMC sampler used by the internal nodes. It contains n_particles particles and the weights are instanciated to
    uniform probability (1/n_particles).
    
    Attributes
    ----------
    - n_particles, int, number of particles
    - log_Z_hat, float, estimated log evidence
    - particles, list(Particle), a list containing the particles (as Particle objects)
    - probabilities, list or array, contains the weights associated to the sampler.
    """
    def __init__(self, n_particles) : 
        self.log_Z_hat = 0.0
        self.n_particles = n_particles
        self.particles = [0]*n_particles
        self.probabilities = 1/(np.ones(n_particles)*n_particles)
        
    def sample(self) : 
        index = np.random.choice([i for i in range(len(self.particles))], size = 1, p = self.probabilities)[0]
        return self.particles[index]
    
    def sample_logdensity(self) : 
        index = np.random.choice([i for i in range(len(self.particles))], size = 1, p = self.probabilities)[0]
        return self.particles[index].log_density()
    

class LeafParticleApprox :
    """
    SMC sampler used by the leaf nodes. It contains n_particles particles and the weights are instanciated to
    zero.
    
    Attributes
    ----------
    - n_particles, int, number of particles
    - log_Z_hat, float, estimated log evidence
    - particles, list(Particle), a list containing the particles (as Particle objects)
    - probabilities, list or array, contains the weights associated to the sampler.
    - y, dictionary, contains the information of the node (successes and trials)
    """
    def __init__(self, n_particles, node_info) : 
        self.log_Z_hat = 0.0
        self.n_particles = n_particles
        self.y = node_info
        self.particles = []
        self.probabilities = np.zeros(n_particles)
        #if initialization : 
        #    for i in range(n_particles) : 
        #        self.particles.append(Particle())
    
    @staticmethod
    def beta_proposal(alpha, beta) : 
        """Return a sample of a Beta distribution with parameters (alpha, beta)"""
        return np.random.beta(alpha,beta) 

    def sample(self) :
        """
        For each particle in the system, draw a value p from the beta distribution (1+m, 1+(M-m)), 
        calculates log_pi as the log density of a Binomial (M,p) at point m, and replace the particle with a
        a new one that will be used by internal nodes. The new particle has a NormalMessagePassing message 
        as internal nodes are multivariate normals. The log density of the particle is log_pi.
        """
        for i in range(self.n_particles) :
            
            proposal = self.beta_proposal(1+self.y['successes'], 1+ (self.y['trials'] - self.y['successes'])) 
            log_pi = np.log(comb(self.y['trials'], self.y['successes'])) + self.y['successes'] * np.log(proposal) + (self.y['trials']-self.y['successes']) * np.log(1-proposal)
            logit = transform(proposal)
            leaf = NormalMessagePassing([logit], 0.0, 0)
            self.particles.append(Particle(self.y, leaf, log_pi, descendant_variance=0))
            
        return self
    

    
    