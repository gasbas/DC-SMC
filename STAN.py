import pystan
import numpy as np
import matplotlib.pyplot as plt

dataDeclaration = []
distributions = []
parametersDeclarations = []
data = []
already_done = []
already_done_declaration = []

 

def readData_recur(tree, node, parent) : 
    """
    Create the 3 parts for the initialisation of the model in STAN the by induction
    
    ----------
    Params
    tree : NetworkX tree
    node : str
    parent : str or Null
    ----------
    Returns
    dataDeclaration : list
    distributions : list
    parametersDeclarations : list
    data : list
    """
    if parent!=None : 
        varianceVarName = "var_" + parent
        if parent not in already_done :
            parametersDeclarations.append("real<lower=0> " + varianceVarName + ";") 
            if parent not in already_done_declaration : 
                parametersDeclarations.append("real a_" + parent + ";")
            distributions.append("var_" + parent + " ~ exponential(1);")
            already_done.append(parent)

        parametersDeclarations.append("real a_" + node + ";")
        already_done_declaration.append(node)

        distributions.append("a_" + node + " ~ " + "normal( a_" + parent + ", " + varianceVarName + ");")

    if tree.get_child(node)==[] :
        numberOfSuccessVar = 'a_' + node + "_numberOfSuccesses"
        numberOfTrialsVar = 'a_' + node + "_numberOfTrials"
        dataDeclaration.append("int<lower=0> " + str(numberOfSuccessVar) + ";")
        dataDeclaration.append("int<lower=0> " + str(numberOfTrialsVar) + ";")
        distributions.append("" + str(numberOfSuccessVar) + " ~ binomial_logit(" + str(numberOfTrialsVar) + ", a_" + node + ");")
        node_value = tree.T.nodes[node]
        data.append(node_value['successes'])
        data.append(node_value['trials'])

    else : 
        for child in tree.get_child(node) :
            readData_recur(tree, child, node)
    
    return dataDeclaration, distributions, parametersDeclarations, data



class model_STAN : 
    
    def __init__(self, tree) : 
        self.dataDeclaration, self.distributions, self.parametersDeclarations, self.data = readData_recur(tree, 'root', None)
                  
                
    def create_model_file(self) : 
        
        modelFile = open("model.stan",'w')
        
        out = "data { \n"
        for line in self.dataDeclaration : 
            out += "  " + str(line) + " \n"
        out += "} \n"
        
        out += "parameters { \n"
        for line in self.parametersDeclarations : 
            out += "  " + str(line) + " \n"
        out += "} \n"
        
        out += "model { \n"
        for line in self.distributions : 
            out += "  " + str(line) + " \n"
        out += "} \n"
        
        modelFile.write(out)
        modelFile.close()
        
        
    def create_data_dict(self) :
        
        map_data = [""] * len(self.dataDeclaration)
        
        for i in range(len(self.dataDeclaration)) : 
            map_data[i] = self.dataDeclaration[i][13:-1]
        return dict(zip(map_data, data))
        
    def init_model(self): 
        
        self.create_model_file()
        sm = pystan.StanModel("model.stan")
        return sm
    
    def fit(self, iterations=100, chains=4, thin=10): 
        sm = self.init_model()
        data_dict = self.create_data_dict()
        fit = sm.sampling(data=data_dict, iter=iterations, chains = chains, thin = thin)
        return fit 
    
def stan_plot(var_dfs, sample_sizes) :

    var_list = [elt for elt in list(fit_df.columns) if elt[:3] == 'var' and len(elt) < 12]
    
    colors = ['orange','green','grey', 'blue', 'red']
    max_samples = len(colors)
              
    nb_samples = len(var_dfs)  
    n_cols = len(var_list)
    fig,ax = plt.subplots(nb_samples, n_cols, figsize = (12,6), sharex=True, sharey=True)
    
    if not isinstance(var_dfs, list) :  
        var_dfs = [var_dfs]
        
    elif len(var_dfs) > max_samples :
        var_dfs = var_dfs[:max_samples]
        sample_sizes = sample_sizes[:max_samples]
        
    for idx in range(0, nb_samples) : 
        ax[idx][0].set_ylabel('N = ' + sample_sizes[idx], fontsize = 12)
        for col in range(0, n_cols) : 
            ax[idx][col].grid(visible = True)
            param = var_dfs[idx][var_list[col]]
            sns.kdeplot(param, color = colors[idx], lw = 1.5, ax =ax[idx][col], fill = True)
      
    ax[-1][0].set_xlabel('DC' , fontsize = 12)
    for col in range(1, n_cols):
        ax[-1][col].set_xlabel('Ward ' + str(col) , fontsize = 12)
    plt.show()

