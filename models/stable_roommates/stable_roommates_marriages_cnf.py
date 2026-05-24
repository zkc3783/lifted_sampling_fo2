import numpy as np
from sys import argv
from random import randint
from subprocess import run, TimeoutExpired, CalledProcessError
from itertools import product
from pysat.card import *

class cnf_problem:

    def __init__(self, NEG_INF):
        self.next_var = 1
        self.vars = dict()
        self.body = []
        
        #this is used to denote impossible pairs in stable marriages
        #set this to the lowest value for which pairs are impossible
        self.NEG_INF = NEG_INF

    def to_file(self, fname):
        with open(fname, "w") as f:
            f.write(f"p cnf {self.next_var - 1} {len(self.body)}\n")
            for clause in self.body:
                out = " ".join(str(x) for x in clause)
                f.write(f"{out} 0\n")

    def create_pairs(self, agents):
        n_agents = sum(agents)
        self.vars["pair"] = np.arange(n_agents**2, dtype=int).reshape(n_agents, n_agents) + self.next_var
        self.next_var += n_agents**2

        for i in range(n_agents):
            self.body.append([-self.vars["pair"][i, i]])
            self.body.append(self.vars["pair"][i, :])
            for j in range(n_agents):
                if i == j:
                    continue
                self.body.append([-self.vars["pair"][i, j], self.vars["pair"][j, i]])
                for k in range(j + 1, n_agents):
                    self.body.append([-self.vars["pair"][i, j], -self.vars["pair"][i, k]])

    def forbid_pairs(self, i, j, k, l, agents_cumsum):
        for i_prime in range(agents_cumsum[i], agents_cumsum[i + 1]):
            for j_prime in range(agents_cumsum[j], agents_cumsum[j  + 1]):
                for k_prime in range(agents_cumsum[k], agents_cumsum[k + 1]):
                    for l_prime in range(agents_cumsum[l], agents_cumsum[l + 1]):
                        if i_prime == k_prime and j_prime == l_prime:
                            continue
                        self.body.append([-self.vars["pair"][i_prime, j_prime], -self.vars["pair"][k_prime, l_prime]])

    def forbid_same_sex(self, preference_graph, agents):
        agents_cumsum = [0]
        for n_agents in agents:
            agents_cumsum.append(agents_cumsum[-1] + n_agents)

        for i in range(preference_graph.shape[0]):
            for j in range(i + 1, preference_graph.shape[1]):
                if preference_graph[i, j] > self.NEG_INF:
                    continue
                for i_prime in range(agents_cumsum[i], agents_cumsum[i + 1]):
                    for j_prime in range(agents_cumsum[j], agents_cumsum[j + 1]):
                        self.body.append([-self.vars["pair"][i_prime, j_prime]])
                        self.body.append([-self.vars["pair"][j_prime, i_prime]])

    def encode_stability(self, preference_graph, agents):
        
        agents_cumsum = [0]
        for n_agents in agents:
            agents_cumsum.append(agents_cumsum[-1] + n_agents)
        for i in range(preference_graph.shape[0]):
            for j in range(i, preference_graph.shape[1]):
                for k in range(i, preference_graph.shape[0]):
                    for l in range(k, preference_graph.shape[1]):
                        if i == k and l < j:
                            continue
                        if preference_graph[i, j] <= self.NEG_INF or preference_graph[j, i] <= self.NEG_INF \
                            or preference_graph[l, k] <= self.NEG_INF or preference_graph[k, l] <= self.NEG_INF:
                            continue
                        if ((preference_graph[i, j] < preference_graph[i, k] and preference_graph[k, i] > preference_graph[k, l]) or
                            (preference_graph[j, i] < preference_graph[j, k] and preference_graph[k, j] > preference_graph[k, l]) or
                            (preference_graph[i, j] < preference_graph[i, l] and preference_graph[l, i] > preference_graph[l, k]) or
                            (preference_graph[j, i] < preference_graph[j, l] and preference_graph[l, j] > preference_graph[l, k])):
                                self.forbid_pairs(i, j, k, l, agents_cumsum)
    

if __name__ == "__main__":
    
    #stm_5_X.wfomcs / cnf
    preference_graph = np.array([[-1, -1, 3, 2, 1], 
                                 [-1, -1, 2, 3, 1], 
                                 [2, 1, -1, -1, -1], 
                                 [1, 2, -1, -1, -1], 
                                 [1, 2, -1, -1, -1]]) 

    agent_counts = [8,8,5,6,5] 
    # #12:33222
    # #16:44233
    # #20:55433
    # #24:66444
    # #28:77545
    # #32:88565

    # stm_4_X.wfomcs / cnf
    # preference_graph = np.array([[-1, -1, 2, 1],  
    #                              [-1, -1, 3, 1], 
    #                              [1, 2, -1, -1], 
    #                              [1, 2, -1, -1]])                             
    # agent_counts = [12,12,12,12]
    # #20:5555
    # #24:6666
    # #28:7777
    # #32:8888
    # #36:9999
    # #40:10101010
    # #48:12121212


    domain_size=sum(agent_counts)
    fname = f"stm_5_{domain_size}.cnf"
    
    #this is used to denote impossible pairs in stable marriages
    #set this to the lowest value for which pairs are impossible
    NEG_INF = -1

    problem = cnf_problem(NEG_INF=NEG_INF)
    problem.create_pairs(agent_counts)
    problem.forbid_same_sex(preference_graph, agent_counts)
    problem.encode_stability(preference_graph, agent_counts)
    problem.to_file(fname)

   