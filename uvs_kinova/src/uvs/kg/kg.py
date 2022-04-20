"""
Knowledge Graph Utilities

"""

import numpy as np
import graphviz

class Triple(object):
    """Base class for a triple of (subject, predicate, object).
    """
    def __init__(
        self,
        subject: str,
        predicate: str,
        object: str,
        ):
        self.subject = subject
        self.predicate = predicate
        self.object = object

    def __str__(self):
        return "{} {} {}".format(self.subject, self.predicate, self.object)


class KnowledgeGraph(object):
    """Base class for a knowledge graph.
    """
    def __init__(
        self,
        ):
        self.triples = []   # Hold triples
    
    # ####################
    # Some base properties
    # ####################
    def add(
        self,
        subject: str,
        predicate: str,
        object: str,
        ):
        assert subject is not None and len(subject) > 0
        assert predicate is not None and len(predicate) > 0
        assert object is not None and len(object) > 0

        triple = Triple(subject, predicate, object)
        self.add_triple(triple)

    def add_triple(
        self,
        triple: object,
        ):
        if triple not in self.triples:
            self.triples.append(triple)

    def remove_triple(
        self,
        triple: object,
        ):
        """Remove one triple from KG.
        """
        self.triples.remove(triple)

    def remove_entity(
        self, 
        entity: str,
        ):
        """Remove one or more triplets with the specified entity.
        """
        self.triples = [triple for triple in self.triples if triple.subject != entity and triple.object != entity]

    def remove_relation(
        self, 
        relation: str,
        ):
        """Remove one or more triplets with the specified relation / predicate.
        """
        self.triples = [triple for triple in self.triples if triple.predicate != relation]

    def clear(self):
        self.triples = []

    def __str__(self):
        triples_str = ''
        for triple in self.triples: 
            triples_str += str(triple) + "\n"
        return triples_str.strip()

    def get_entities(self):
        """Return all entities (subjects & objects) held in the KG.
        """
        entities = set()
        for triple in self.triples:
            entities.add(triple.subject)
            entities.add(triple.object)
        return entities

    def get_relations(self):
        """Return all relations / predicates held in the KG.
        """
        relations = set()
        for triple in self.triples:
            relations.add(triple.predicate)
        return relations

    def to_graphviz(
        self,
        engine='neato',
        edge_len=1.75,
        edgecolor='#23a6db66',
        fillcolor='lightblue',
        ):
        """Convert KG object into an graphviz.DiGraph for visualization.
        Returns:
            g: graphviz.Digraph.
        """
        def color_change(text):
            text = text.lower()
            if 'eef' in text:
                return "#e6def7", 'mediumpurple'
            elif 'goal' in text:
                return "#98FB98", '#32CD32'
            else:
                return 'lightblue', '#23a6db66'

        g = graphviz.Digraph(engine=engine)
        g.attr('graph', 
               center='true',
               fontname='Helvetica',
               bgcolor='transparent',
               )
        g.attr('node', 
               shape='circle', 
               fixedsize='true', width='0.75', height='0.75',
               color=edgecolor, fillcolor=fillcolor, style='filled', 
               fontcolor='black', fontsize='12', fontname='Helvetica',
               )
        g.attr('edge', 
               penwidth='1.5', color='grey', 
               fontname='Helvetica', fontsize='11', fontcolor='#474747',
               len=str(edge_len),
               )
        for triple in self.triples:
            fillcolor, edgecolor = color_change(triple.subject)
            g.attr('node', fillcolor=fillcolor, color=edgecolor)
            g.node(triple.subject)

            fillcolor, edgecolor = color_change(triple.object)
            g.attr('node', fillcolor=fillcolor, color=edgecolor)
            g.node(triple.object)

            g.edge(triple.subject, triple.object, label=triple.predicate)
        return g


    # ####################
    # NOTE: Add Entity Linking specifics below in future
    # TODO: Link entities in KG to DBpedia
    # ####################