#!usr/bin/python3.10
from sys import argv
import numpy as np

#   IMPLEMENTATION DETAILS:
#   Generates .wfomcs file for stable roommates or stable marriages problem
#   The generated file counts each permutation of classes as a separate model
#   i. e. {PairedWithAB(1), PairedWithAB(2), paired(1, 2)} is two models
#   (as 1 could be class A or B and 2 is the other class) and also 
#   {PairedWithAB(1), PairedWithAB(2), paired(1, 2), paired(2, 1), 
#    PairedWithCD(3), PairedWithCD(4), paired(3, 4), paired(4, 3)}
#   will be counted as 4 models and
#   {PairedWithAB(3), PairedWithAB(4), paired(3, 4), paired(4, 3),
#    PairedWithCD(1), PairedWithCD(2), paired(1, 2), paired(2, 1)}
#   will be counted as another 4 models.
#   If this behaviour is not desired and unary evidence is cheap, let me know on Discord and
#   I will provide another implementation.
#   Also, this can be accounted for by dividing the result by |V|! / (|A|! * |B|! * ... * |N|!),
#   where |V| is the domain size and |A|, |B|, ..., |N| is the agent counts.
#   (I might be wrong about the result correction, but on small examples with 2 classes it works).


#Set negative infinity in preference graph
#Negative infinity pairs cannot exist!
#Used in stable marriages, 
#   should be set BELOW minimum preference for stable roommates
NEG_INF = -1

#Name of output .wfomcs file
F_OUT = "stable_roommates.wfomcs"

def create_class_clauses(n_classes, preference_graph):
    if n_classes > 26:
        print("Too many classes")
        return None
    out = []
    base = ord('A')
    classes = [f"PairedWith{chr(i + base)}{chr(j + base)}(X)" \
                for i in range(n_classes) for j in range(i, n_classes)\
                if preference_graph[i, j] > NEG_INF and preference_graph[j, i] > NEG_INF]
    clause = "\\forall X: (" + " | ".join(classes) + ")"
    out.append(clause)
    for i in range(n_classes):
        for j in range(i, n_classes):
            if preference_graph[i, j] <= NEG_INF or \
                preference_graph[j, i] <= NEG_INF: 
                continue
            for k in range(i, n_classes):
                for l in range(k, n_classes):
                    if preference_graph[k, l] <= NEG_INF or \
                        preference_graph[l, k] <= NEG_INF:
                        continue
                    if k == i and l <= j:
                        continue
                    clause = f"\\forall X: (~PairedWith{chr(i + base)}{chr(j + base)}(X) | ~PairedWith{chr(k + base)}{chr(l + base)}(X))"
                    out.append(clause)
    return out

def create_pairs(n_classes):
    out = []
    if n_classes > 26:
        print("Too many classes")
        return None
    base = ord('A')
    clause = f"\\forall X: (\\forall Y: (" + "paired(X, Y) -> paired(Y, X)" + "))"
    out.append(clause)
    clause = "\\forall X: (\\exists_{=1} Y: (paired(X, Y)))"
    out.append(clause)
    clause = f"\\forall X: (~(paired(X, X)))"
    out.append(clause)
    for i in range(n_classes):
        for j in range(i, n_classes):
            if preference_graph[i, j] <= NEG_INF or \
                preference_graph[j, i] <= NEG_INF: 
                continue
            clause = f"\\forall X: (\\forall Y: (~paired(X, Y) | ~PairedWith{chr(i + base)}{chr(j + base)}(X) | PairedWith{chr(i + base)}{chr(j + base)}(Y) ))"
            out.append(clause)
    return out

def create_stability(preference_graph):
    if preference_graph.shape[0] > 26:
        print("Too many classes")
        return None
    if preference_graph.shape[0] != preference_graph.shape[1]:
        print("Preference graph must be a square")
        return None
    out = []
    ccs_out = []
    base = ord('A')
    for i in range(preference_graph.shape[0]):
        for j in range(i, preference_graph.shape[1]):
            if preference_graph[i, j] <= NEG_INF or \
                preference_graph[j, i] <= NEG_INF: 
                continue
            for k in range(i, preference_graph.shape[0]):
                for l in range(k, preference_graph.shape[1]):
                    if preference_graph[k, l] <= NEG_INF or \
                        preference_graph[l, k] <= NEG_INF: 
                        continue
                    if i == k and l <= j:
                        continue
                    print(f"i: {chr(i + base)}, j: {chr(j + base)}, k: {chr(k + base)}, l: {chr(l + base)}", end=" ")
                    if ((preference_graph[i, j] < preference_graph[i, k] and preference_graph[k, i] > preference_graph[k, l]) or
                        (preference_graph[j, i] < preference_graph[j, k] and preference_graph[k, j] > preference_graph[k, l]) or
                        (preference_graph[i, j] < preference_graph[i, l] and preference_graph[l, i] > preference_graph[l, k]) or
                        (preference_graph[j, i] < preference_graph[j, l] and preference_graph[l, j] > preference_graph[l, k])):
                        print("is blocking")
                        clause = f"\\forall X: (\\forall Y: (~PairedWith{chr(i + base)}{chr(j + base)}(X) | ~PairedWith{chr(k + base)}{chr(l + base)}(Y) | paired(X, Y)))"
                        out.append(clause)
                    else:
                        print("is not blocking")

    return out, ccs_out

def unary_evidence_clauses(preference_graph, agents):
    out = []
    base = ord('A')
    for i in range(len(agents)):
        if preference_graph[i, i] > NEG_INF:
            out.append(f"\\forall X: (~PairedWith{chr(i + base)}{chr(i + base)}(X) | {chr(base + i)}(X))")
        lpart = " | ".join(f"PairedWith{chr(j + base)}{chr(i + base)}(X)" for j in range(i) if preference_graph[i, j] > NEG_INF and preference_graph[j, i] > NEG_INF)
        selfpart = f"PairedWith{chr(i + base)}{chr(i + base)}(X)" if preference_graph[i, i] > NEG_INF else ""
        rpart = " | ".join(f"PairedWith{chr(i + base)}{chr(j + base)}(X)" for j in range(i + 1, len(agents)) if preference_graph[i, j] > NEG_INF and preference_graph[j, i] > NEG_INF) if i < len(agent_counts) - 1 else ""
        out.append(f"\\forall X: (~{chr(base + i)}(X) | " + " | ".join(x for x in (lpart, selfpart, rpart) if x != "") + ")")
        for j in range(i + 1, len(agents)):
            if preference_graph[i, j] > NEG_INF and preference_graph[j, i] > NEG_INF:
                out.append(f"\\forall X: (~PairedWith{chr(i + base)}{chr(j + base)}(X) | {chr(base + i)}(X))")
                out.append(f"\\forall X: (~PairedWith{chr(i + base)}{chr(j + base)}(X) | {chr(base + j)}(X))")
    return out



if __name__ == "__main__":

    #preference graph : sets preference towards any class
    #generalizes specifying order in the original problem 
    #   if no two preferences for each class are the same
    #negative preferences work, but must be HIGHER than NEG_INF paramter 
    #   (reserved for impossible pairs, can be set above)
    #use NEG_INF preferences in stable marriages to differentiate men and women

    #ENCODING = "ccs"
    ENCODING = "evidence"

    # preference_graph = np.array([[2, 1, 0, 4], 
    #                              [1, 4, 3, 0], 
    #                              [0, 3, 2, 1], 
    #                              [0, 2, 1, 4]]) #example preference graph for stable roommates
    '''stm_{sum(agent_counts)}_5_wfomcs''' 
    preference_graph = np.array([[-1, -1, 3, 2, 1], 
                                 [-1, -1, 2, 3, 1], 
                                 [2, 1, -1, -1, -1], 
                                 [1, 2, -1, -1, -1], 
                                 [1, 2, -1, -1, -1]]) 

    agent_counts = [3,3,2,2,2] 

    domain_size = sum(agent_counts)

    if domain_size % 2 == 1:
        raise RuntimeError("DOMAIN CANNOT BE ODD SIZE!")

    clauses = []
    ccs = []
    new_clauses = create_class_clauses(preference_graph.shape[0], preference_graph)
    for clause in new_clauses:
        clauses.append(clause)
    new_clauses, ccs = create_stability(preference_graph)
    for clause in new_clauses:
        clauses.append(clause)
    new_clauses = create_pairs(preference_graph.shape[0])
    for clause in new_clauses:
        clauses.append(clause)
    if ENCODING == "evidence":
        new_clauses = unary_evidence_clauses(preference_graph, agent_counts)
        for clause in new_clauses:
            clauses.append(clause)
    
    with open(F_OUT, "w") as f:
        
        f.write(" &\n".join(clauses))
        f.write("\n\n")
        if ENCODING == "ccs":
            f.write(f"V = {domain_size}\n")
        elif ENCODING == "evidence":
            f.write("V = {" + ", ".join(f"a{i}" for i in range(sum(agent_counts))) + "}\n")
        f.write("\n")
        base = ord('A')
        for i in range(preference_graph.shape[0]):
            for j in range(i + 1, preference_graph.shape[1]):
                if preference_graph[i, j] <= NEG_INF or preference_graph[j, i] <= NEG_INF:
                    continue
                # 1.414213562 as substitute for sqrt(2)
                # can be represented in WFOMC, but not read from .wfomcs afaik
                f.write(f"1.414213562 1 PairedWith{chr(base + i)}{chr(base + j)}\n")

        #each agent is paired to exactly one other
        #f.write(f"\n|paired| = {domain_size}\n")

        if ENCODING == "ccs":    
            #ccs specifying number of agents in each class
            base = ord("A")
            for i in range(preference_graph.shape[0]):
                line = [f"|PairedWith{chr(base + i)}{chr(base + j)}|" for j in range(preference_graph.shape[0]) \
                        if i < j and preference_graph[i, j] > NEG_INF and preference_graph[j, i] > NEG_INF]
                if preference_graph[i, i] > NEG_INF and preference_graph[i, i] > NEG_INF:
                    line = line + [f"2|PairedWith{chr(base + i)}{chr(base + i)}|"]
                line = line + [f"|PairedWith{chr(base + j)}{chr(base + i)}|" for j in range(preference_graph.shape[0]) \
                                if j < i and preference_graph[i, j] > NEG_INF and preference_graph[j, i] > NEG_INF]
                f.write(" + ".join(line) + f" = {2 * agent_counts[i]} \n")

            for i in range(preference_graph.shape[0]):
                for j in range(i + 1, preference_graph.shape[1]):
                    if preference_graph[i, i] > preference_graph[i, j] and preference_graph[j, j] > preference_graph[j, i]:
                        f.write(f"|PairedWith{chr(base + i)}{chr(base + j)}| <= 2 \n")
        elif ENCODING == "evidence":
            #f.write(", ".join(f"{2}(a{sum(agent_counts[:i]) + cur_agent})" for i, agent_count in enumerate(agent_counts) for cur_agent in range(agent_count)))
            base = ord('A')
            f.write("\n\n")
            f.write(", ".join(f"{chr(base + i)}(a{sum(agent_counts[:i]) + cur_agent})" for i, agent_count in enumerate(agent_counts) for cur_agent in range(agent_count)))
            f.write("\n")