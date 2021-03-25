#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:09:17 2019

This page contains a lot of the code used to build the initial dataset and
graph.  It's not in a specific order for running, but i generally followed
the order seen.  The purpose of the snippets is generally discussed at the top.



"""

import string
from string import digits
import keras
import tensorflow as tf
import json
#import graph_class as gc

import nltk, re
#from nltk.corpus import treebank, stopwords
#from nltk.tokenize import sent_tokenize, word_tokenize
#from nltk import ne_chunk_sents
# nltk.download('maxent_ne_chunker')
nltk.download('punkt')
import requests
import pandas as pd

import pycorenlp
from pycorenlp import StanfordCoreNLP
# import node

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Function Defs

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# For removing punctuation
remove_punc = str.maketrans('', '', string.punctuation)
# For removing digits
remove_digits = str.maketrans('', '', digits)

def prepContent(content):
    
    temp = nltk.sent_tokenize(content)
    
    '''
    Prepping the sentences
    '''
    # Lower-cases
    temp = [x.lower() for x in temp]
    
    # Remove quotes
    temp = [re.sub("'", '', x) for x in temp]
    
    # Remove punctuation / digits
    temp = [x.translate(remove_punc) for x in temp]
    #temp = [x.translate(remove_digits) for x in temp]
    
    # Remove spaces
    temp = [x.strip() for x in temp]
    temp = [re.sub(" +", " ", x) for x in temp] 

    return temp
    


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Parameters

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
topic = "Georgetown_University"





"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Download from Wikipedia

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Page_List = [
'Campuses_of_Georgetown_University',
'Category:Georgetown_University_schools',
'Category:Georgetown_University_programs',
'Category:Georgetown_Hoyas',
'Category:Georgetown_University_student_organizations',
'Category:Georgetown_University_buildings',
'Category:Georgetown_University_publications',
'Category:Georgetown_University_people',
'History_of_Georgetown_University',
'Hoya_Saxa',
'Housing_at_Georgetown_University',
'Category:Georgetown_University_templates',
'Georgetown_University_Library',
'Georgetown_University_Alma_Mater',
'President_and_Directors_of_Georgetown_College',
'Category:Georgetown_University_Medical_Center',
'Energy:_A_National_Issue',
'Center_for_Strategic_and_International_Studies',
'Georgetown_University',
'Georgetown_University_Police_Department',
'1838_Georgetown_slave_sale',
'St._Thomas_Manor',
'Anne_Marie_Becraft',
'List_of_Georgetown_University_commencement_speakers',
'Bishop_John_Carroll_(statue)',
'2019_college_admissions_bribery_scandal'
]



import matplotlib.pyplot as plt
import re

import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('en')

Topic_Dict = {}

########################### TS stuff ###########################
'''
page = 'Category:Georgetown_University_schools'
page = 'Georgetown University Law Center'

current_list = wiki_wiki.page(page)
cats_in_page = current_list.categorymembers
keys = cats_in_page.keys()
'''
################################################################

individual_pages = []

# Recursive method to retrieve all the pages in a category page of wikipedia
# inlcusive of nested category pages.

def get_all_children_pages(categories):
    good_list = []
    input_dict = categories.categorymembers
    keys = input_dict.keys()
    
    for key in keys:
        
        if "Category" in key:
            
            if "alumni" not in key:
                temp = get_all_children_pages(wiki_wiki.page(key))
                good_list += temp
            
        else:
            
            good_list.append(key)
            
    return good_list


# Builds the list
for item in Page_List:
    if "Category" in item:
        output_list = get_all_children_pages(wiki_wiki.page(item))
        individual_pages += output_list
        
    else:
        individual_pages.append(item)

# Puts the pages into a dictionary for easy access.
for page in individual_pages:
    Topic_Dict[page] = wiki_wiki.page(page)
    
    
###############################################################################
#    
##   Building Graph
#
###############################################################################

import networkx as nx

# New Graph
G = nx.Graph()

# Cycle through individual pages associated with Georgetown
page_index = 0


for page in individual_pages:
    
    try:
    
        topic_page = wiki_wiki.page(page)
        
        topic = topic_page.title
        G.add_node(topic, summary = topic_page.summary)
        
        # nx.draw(G, with_labels = True)  
        # G.nodes["Georgetown_University"]
        
        page_links = topic_page.links
        
        page_index += 1
        index = 0
        
        for link in page_links.keys():
            if "Category" not in link:
                new_page = wiki_wiki.page(link)
                G.add_node(new_page.title, summary = new_page.summary)
                G.add_edge(topic, new_page.title)
                index += 1
                if (index % 50 == 0):
                    print("On page " + str(page_index) + " of " + str(len(individual_pages)) + " and processing record " + str(index) + " of " + str(len(page_links.keys())))
        
    except:
        print("Error in download.  Probably Timeout")

        
# len(G)
        
import pickle
# pickle.dump(G, open("Topic_Graph.pkl", "wb"))

import networkx as nx
G = pickle.load(open("./Other-Data/Topic_Graph.pkl", "rb"))
len(G)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Information Extraction Setup

Based on The Stanford NLP Group Open Information Extraction work
https://nlp.stanford.edu/static/software/openie.shtml#Usage

Download from: https://stanfordnlp.github.io/CoreNLP/#download

Put into a folder location you can access and use the following command to
start a server:
    
java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

You can then use the API wrapper from:

https://github.com/smilli/py-corenlp

Install with: 
    pip install pycorenlp
    
If you want to play with it you can go to http://localhost:9000 in a browser

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    


import pycorenlp, os
from pycorenlp import StanfordCoreNLP
from relation_standardizer import Relation_Standardizer
import sys

nlp = StanfordCoreNLP('http://localhost:9000')

rs = Relation_Standardizer('Bob')

#############
#
#  Filling the buffer a bit
#
#############

relation_tuple = []
cleaned_tuples = []
index = 0

for page in individual_pages:
    
    topic_page = wiki_wiki.page(page)
    sentences = prepContent(topic_page.text)
    
    index += 1
    if index % 10 == 0:
        print('Currently processing page: ' + str(index))
    
    for sentence in sentences:
        
        try:
    
            results = nlp.annotate(sentence, properties = {'annotators': 'openie', 'outputFormat': 'json'})
            sub_result = results['sentences'][0]['openie']
            
            for i in range(len(sub_result)):
                relation_tuple.append([sub_result[i]['subject'], sub_result[i]['relation'], sub_result[i]['object']])
                
        except:
            
            print("Something Happened.  Booooooooo!!!!")


for tup in relation_tuple:
    relation = rs.standardize(tup[1])
    
    if relation != 'null':
        cleaned_tuples.append([tup[0], relation, tup[2]])


sys.getsizeof(cleaned_tuples)

import pickle
# pickle.dump(cleaned_tuples, open("cleaned_tuples.pkl", "wb"))

import networkx as nx
cleaned_tuples = pickle.load(open("./Other-Data/cleaned_tuples.pkl", "rb"))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

BERT Setup

We're using a simplified version with BERT set up as a query server. The code 
is available from:

https://github.com/hanxiao/bert-as-service

You still need to download a version of BERT from the primary website as this
just taps into that.

The webpage gives instructions for installation.

Here is the command from MacOS terminal I used to start it up:
bert-serving-start -model_dir ~/BERT-data/uncased_L-12_H-768_A-12/ -num_worker=2

on my Linux box:
bert-serving-start -model_dir ~/Final_Proj/BERT-Data/ -num_worker=4

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 

from bert_serving.client import BertClient
bc = BertClient()

# The results from CoreNLP are from in a dictionary.  Here I have the proper
# parts being saved to sub_result, which ends up being a list of dictionaries
# with each entry in the list (a dict) corresponding to a relationship parse.

# sub_result[0]['subject']
#sub_result[5]['object']

# May use
relation_tuples = []

'''
topic = "Georgetown University"
page = wiki_wiki.page(topic)
'''

# sentences = prepContent(page.text)

temp = bc.encode(["The dog chased the boy ||| The boy got licked by the dog"])

# Cycles through each sentence extracted from the page.
for relation in cleaned_tuples:
    
    # Saves the BERT double encoding to the encoded variable as a 768 length vector
    # Basic idea is to feed these into the actor to see if it can learn to figure
    # out the proper relationships.

    relation_tuples.append([relation[1], bc.encode([relation[0] + ' ||| ' + relation[2]])]) 
    # TODO: Need to include rest of code here.
        

sys.getsizeof(relation_tuples)

import pickle
pickle.dump(relation_tuples, open("relation_tuples.pkl", "wb"))

relation_tuples = pickle.load(open("./Other-Data/relation_tuples.pkl", 'rb'))
# To change a relation to an int for input to an agent        
rs.relation_to_int('nominate')

# To change a result from an agent back to a relation
rs.int_to_relation(0)

node_links = list(G.edges("Anne Speckhard"))

link_predictions = []

import classes

sess = tf.Session()
my_critic = classes.Critic(768, 0.0001, 768, 768)
sess.run(tf.global_variables_initializer())


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This works, but takes too long.  Am going to have to do a bunch of pre-process
work to let this speed up.

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
predictions = []
# For determining the most poorly predicted node from the current node
for link in node_links:
    
    relations = []
    encodes = []

    
    sentences = prepContent(G.nodes[link[1]]['summary'])
    
    for sentence in sentences:
        try:
            results = nlp.annotate(sentence, properties = {'annotators': 'openie', 'outputFormat': 'json'})
            sub_result = results['sentences'][0]['openie']
            for i in range(len(sub_result)):
                relations.append([sub_result[i]['subject'], sub_result[i]['relation'], sub_result[i]['object']])
        except: 
            print("Problem in Core-NLP.  Skipping")
        
    for relation in relations:
        try:
            encodes = bc.encode([relation[0] + " ||| " + relation[2]])
        except:
            print("Problem in bc encoding.  Skipping")
            
    for encode in encodes:
        predictions.append(sess.run(my_critic.layer3, feed_dict = {my_critic.observation: [encode]}))
                    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This is supposed to cycle through every node in the graph and convert the 
attached summary to extracted relations and then to BERT encodings.  These
shouldn't change over the course of updating agents, so hopefully this works.

I'm taking a maximum of 10 relations/embeddings per node to save on time/space
as we have not too much time left.  I figure 10 random nodes should give a 
decent enough measure of how poorly we think we can predict.  In the end we
can just pull the embeddings from each node and run it through our critic.

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
import random
# node = G.nodes["Georgetown University"]
index = 0

thing = G.nodes

for name in G.nodes:
    
    node = G.nodes[name]
    node = G.nodes["Georgetown_University"]
        
    index += 1
    if index % 25 == 0:
        print("Processing node " + str(index) + " of " + str(len(G)))
    
    relations = []
    encodes = []
    
    try:
        sentences = prepContent(node['summary'])
        
        for sentence in sentences:
            try:
                results = nlp.annotate(sentence, properties = {'annotators': 'openie', 'outputFormat': 'json'})
                sub_result = results['sentences'][0]['openie']
                for i in range(len(sub_result)):
                    relations.append([sub_result[i]['subject'], sub_result[i]['relation'], sub_result[i]['object']])
            except: 
                print("Problem in Core-NLP.  Skipping")
            
        if len(relations) > 10:
            relations = random.sample(relations, 10)
            
        for relation in relations:
            try:
                encodes.append(bc.encode([relation[0] + " ||| " + relation[2]]))
            except:
                print("Problem in bc encoding.  Skipping")
    except:
        print("Problem in Summary.  Leaving empty.")
        
            
    node['relations'] = relations
    node['encodes'] = encodes
    
    node['relations']
    
    G.nodes["Georgetown_University"]
    
# Saving the data for later use    
pickle.dump(G, open("G_augmented.pkl", "wb")) 

# To load the data   
G = pickle.load(open("./Other-Data/G_augmented.pkl", 'rb'))    