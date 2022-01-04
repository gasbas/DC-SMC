import pandas as pd
import numpy as np
import json
import scipy
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import networkx as nx


def build_tree(df) : 
    """
    Builds a NetworkX tree (directed graph from root to leaves) based on hierarchical data. Data is supposed to be
    sorted from hierarchical order where the first column should be the first hierarchical order.
    The last column is supposed to be the number of succeses and the second last is supposed to be the number of 
    trials.
    ----------
    Params
    df: pandas.DataFrame
    ----------
    Returns
    (networkx.DiGraph, int, int): the created tree T, the total number of test trials and the total number of succeses.
    """
    
    T = nx.DiGraph()
    T.add_node('root')
    T.nodes['root']['level'] = 0
    n_obs=0
    n_successes = 0
    n_total_attempts = 0
    
    n_levels = len(df.iloc[:,:-2].columns) + 1
    
    for idx in range(df.shape[0]) : 
        parent = 'root' 
        curr = df.iloc[idx,:-2]
        for level in range(n_levels) : 
            name = str(parent)+'_'+str(curr.iloc[0])
  
            T.add_node(name)
            T.add_edge(parent, name)
            T.nodes[name]['level'] = level + 1
            parent = name
            if curr.shape[0] > 1 : 
                curr = curr.iloc[1:]
            else : 
                T.nodes[name]['trials'] = df.iloc[idx,-2]
                T.nodes[name]['successes'] = df.iloc[idx,-1]
                n_total_attempts += T.nodes[name]['trials']
                n_successes += T.nodes[name]['successes']
                n_obs += 1
            
                
    print(f'Number of hierarchical levels: {n_levels}')
    print(f'Number of leaf nodes: {n_obs}')
    print(f'Number of test instances in the dataset: {n_total_attempts}')
    print(f'Number of successes in the dataset: {n_successes}')
    
    return T, n_obs, n_total_attempts, n_successes

class TreeDataset : 
    """
    A wrapper to access useful methods specifics to trees. Builds a networkX tree (Directed graph from to the root
    down to the leaves) based on hierarchical data.Data is supposed to be
    sorted from hierarchical order where the first column should be the first hierarchical order.
    The last column is supposed to be the number of succeses and the second last is supposed to be the number of 
    trials.
    ----------
    Init
    data_path: str, path to the hierarchical data stored in a csv file.
    ----------
    
    """
    def __init__(self, data_path) : 
        df = pd.read_csv(data_path)
        self.T, self.n_leaf_nodes, self.n_attempts, self.n_successes_ = build_tree(df)
    
    def get_root(self) : 
        return [x for x in self.T.nodes() if self.T.out_degree(x)>=1 and self.T.in_degree(x)==0]
    
    def get_parents(self, node_name) : 
        
        return [i for i in self.T.predecessors(node_name)]
    
    def get_child(self, node_name) :
        return [i for i in self.T.successors(node_name)]
    
    def is_leaf(self, node_name) : 
        if len(self.get_parents(node_name)) == 0 : 
            return True
        else : 
            return False
    def is_root(self, node_name) : 
        if len(self.get_child(node_name)) == 0 : 
            return True
        else : 
            return False
        
    def get_leafs(self) : 
        return [x for x in self.T.nodes() if self.T.out_degree(x)==0 and self.T.in_degree(x)==1]
    
    def recur_order(self, node_name, result) : 
        
        for child in self.get_child(node_name) : 
            self.recur_order(child, result)
        result.append(node_name)
        
    def order_nodes(self) : 
        result  = []
        self.recur_order('root', result)
        return result
    
    def plot(self, show_label = False, save_path = None) : 
        plt.figure(figsize=(16,10))
        plt.axis('off')
        pos = graphviz_layout(self.T, prog="dot")
        
        #nx.draw(T, pos)
        v=nx.draw_networkx_nodes(self.T,pos, node_color = 'grey', alpha = 0.7)
        v=nx.draw_networkx_edges(self.T,pos,width = 0.8, alpha = 0.7)
        if show_label : 
            v=nx.draw_networkx_labels(self.T,pos)
        if save_path : 
            plt.savefig(save_path, dpi = 300)