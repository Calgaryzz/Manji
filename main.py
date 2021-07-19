import graphviz

from importer import construct_from_tfst

sentence_number = 326
starting_node = construct_from_tfst("/home/calgaryzz/workspace/Unitex-GramLab/Unitex/French/Corpus/80jours_snt/text.tfst", sentence_number)