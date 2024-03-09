import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np
from chempy import ReactionSystem, Substance, Reaction
from chempy.kinetics.ode import get_odesys


class System: 

    def __init__(self): 
        self.reactions = [
                Reaction({'A': 1}, {'B': 1}, rate=1e-4),
                Reaction({'B': 1}, {'C': 1}, rate=1e-2),
                Reaction({'C': 1}, {'D': 1}, rate=1e-3),
                Reaction({'D': 1}, {'E': 1}, rate=1e-5),
                Reaction({'E': 1}, {'F': 1}, rate=1e-3),
                Reaction({'F': 1}, {'G': 1}, rate=1e-2),
                Reaction({'G': 1}, {'A': 1,},rate=1e-3)
        ]
        self.substances = [Substance(s) for s in 'ABCDEFG']
        self.reaction_system = ReactionSystem(self.reactions, self.substances)

    def network(self):
        for i in self.reaction_system.rxns: 
            for j, stoich in i.reactants.items(): 
                for product, _ in i.products.items(): 
                    self.graph.add_edge(j.name, product.name)

    def plot(self): 
        self.graph = nx.DiGraph()
        plt.figure(figsize=(20, 20))
        p = nx.circular_layout(self.graph)
        nx.draw(self.graph,
                 p, 
                 node_size=800, 
                 with_labels=True, 
                 node_color='skyblue',
                 font_size=12
                 )
        plt.title('Reaction Network')
        plt.show() 

    def handlers(self): 
        print('Degree per Node:')
        for node, degree in self.graph.degree():
            print(f"{degree}, {node}")
        print("\n-----------------------------------")
        print("\n Betweenness Centrality:")
        b = nx.betweenness_centrality(self.graph)
        for c, node in b.items(): 
            print(f"{c}, {node}")
    
    def analyze(self):
        ode, e = get_odesys(self.reaction_system)
        t, y = ode.integrate(np.linespace(0, 10, 100), np.random.rand(len(self.substances)))
        plt.figure(figsize=(20, 20))
        for i, s in enumerate(self.substances):
            plt.plot(t, y[:, i], label=s.name)
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title('Concentration over Time')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    op = System()
    op.plot()
    op.handlers()
    op.analyze()