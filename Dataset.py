import pandas as pd
import numpy as np
import json
import scipy
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import networkx as nx


def build_tree(df) : 

    T = nx.DiGraph()
    T.add_node('root')
    n_obs=0
    n_successes = 0
    n_total_attempts = 0

    for ward in df['ward_id'].unique() : 
        T.add_node(str(ward))
        T.add_edge('root', str(ward))
        tmp_final = df[df['ward_id']==ward]
            
        for s_id in tmp_final['school_id'].unique() :
            
            name2 = str(ward) + '_'+str(s_id)
            T.add_node(name2)
            T.add_edge(str(ward), name2)
            tmp_final2 = tmp_final[tmp_final['school_id']==s_id]
                
            for year in tmp_final2['year'].unique() :
                
                name3 = name2 + '_'+str(n_obs)
                
                T.add_node(name3)
                successes = tmp_final2[tmp_final2['year']==year]['successes'].values[0]
                T.nodes[name3]['successes'] = successes
                n_successes += successes
                
                attempts = tmp_final2[tmp_final2['year']==year]['trials'].values[0]
                T.nodes[name3]['trials'] = attempts
                n_total_attempts+= attempts
                T.add_edge(name2, name3)
                
                n_obs+=1
                
    print(f'Number of leaf nodes: {n_obs}')
    print(f'Number of test instances in the dataset: {n_total_attempts}')
    print(f'Number of successes in the dataset: {n_successes}')
    
    return T, n_obs, n_total_attempts, n_successes



class TreeDataset : 
    
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
    
    def plot(self) : 
        plt.figure(figsize=(16,10))
        pos = graphviz_layout(self.T, prog="dot")
        #nx.draw(T, pos)
        v=nx.draw_networkx_nodes(self.T,pos)
        v=nx.draw_networkx_edges(self.T,pos)
        v=nx.draw_networkx_labels(self.T,pos)
