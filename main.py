from data_creation.utils import construct_corpus, construct_from_tfst
import networkx as nx

sentence_number = 6
lattice = construct_from_tfst("/home/calgaryzz/workspace/Unitex-GramLab/Unitex/French/Corpus/80jours_snt/text.tfst", sentence_number, visual=True)
sorted = list(nx.topological_sort(lattice))
