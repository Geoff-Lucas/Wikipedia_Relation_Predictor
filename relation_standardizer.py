#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:35:49 2019

@author: geoffreylucas
"""

class Relation_Standardizer:
    
    def __init__(self, name):
        
        self.name = name
        
        self.is_list =          ['is', 'is a', 'was', 'were', 'are', 'been', 'be']
        self.possess_list =     ['has', 'had', 'possess', 'possesses'] 
        self.location_list =    ['location', 'located', 'venue', 'is in', 'place held', 'based in', 'neighborhood']
        self.transform_list =   ['became', 'transform']
        self.served_as_list =   ['served as']
        self.is_with_list =     ['is with', 'was with']
        self.nominated =        ['nominate']
        self.include =          ['include']
        self.win =              ['won', 'wins', 'win', 'awarded', 'award']
        self.received =         ['received', 'got']
        self.play =             ['played', 'plays']
        self.scored =           ['scored', 'score']
        self.named =            ['named']
        self.made =             ['made']
        self.attend =           ['attended', 'attends', 'attend']
        self.represent =        ['represents', 'represented', 'represent']
        self.start =            ['began', 'started', 'start']
        self.finish =           ['finished', 'finish', 'ended']
        self.earned =           ['earned', 'earn']
        self.joined =           ['joined', 'members', 'member']
        self.born =             ['born', 'birth']
        self.led =              ['led']
        self.follow =           ['follow', 'followed', 'follows']
        self.see =              ['see', 'saw']
        self.comparison =       ['as', 'greater than', 'less than']
        
        
        self.relations_list = ['is', 'possesses', 'location', 'transform', 'served_as', 'is_with', 'nominate', 'include', 'win', 'received', 'play', 'scored', 'named',
                               'made', 'attend', 'represent', 'start', 'finish', 'earned', 'joined', 'born', 'led', 'follow', 'see', 'comparison']
        
        
    def standardize(self, relation):
        
        # Note: For some of the relations I'm thinking are going to be very
        # common and part of other words, I'm using the == equality check.
        
        
        response = 'null'
        
        if any(to_check == relation for to_check in self.is_list):
            response = 'is'
            
        if any(to_check == relation for to_check in self.possess_list):
            response = 'possesses'           
            
            
        if any(to_check in relation for to_check in self.location_list):
            response = 'location'
            
        if any(to_check in relation for to_check in self.transform_list):
            response = 'transform' 
        
        
        
        if any(to_check in relation for to_check in self.served_as_list):
            response = 'served_as'
            
        if any(to_check in relation for to_check in self.is_with_list):
            response = 'is_with'         
        
        
        
        if any(to_check in relation for to_check in self.nominated):
            response = 'nominate'
            
        if any(to_check in relation for to_check in self.include):
            response = 'include'  
            
            
            
        if any(to_check == relation for to_check in self.win):
            response = 'win'
            
        if any(to_check == relation for to_check in self.received):
            response = 'received'        
            
            
            
        if any(to_check == relation for to_check in self.play):
            response = 'play'
            
        if any(to_check in relation for to_check in self.scored):
            response = 'scored'    
            
            
            
        if any(to_check in relation for to_check in self.named):
            response = 'named'
            
        if any(to_check == relation for to_check in self.made):
            response = 'made'              
            
            
            
        if any(to_check in relation for to_check in self.attend):
            response = 'attend'
            
        if any(to_check in relation for to_check in self.represent):
            response = 'represent'              



        if any(to_check in relation for to_check in self.start):
            response = 'start'
            
        if any(to_check in relation for to_check in self.finish):
            response = 'finish'    
            
            
            
        if any(to_check in relation for to_check in self.earned):
            response = 'earned'
            
        if any(to_check in relation for to_check in self.joined):
            response = 'joined'              



        if any(to_check in relation for to_check in self.born):
            response = 'born'
            
        if any(to_check == relation for to_check in self.led):
            response = 'led'                
            
            
            
        if any(to_check in relation for to_check in self.follow):
            response = 'follow'
            
        if any(to_check == relation for to_check in self.see):
            response = 'see'              



        if any(to_check == relation for to_check in self.comparison):
            response = 'comparison'           
            
            
        return response
    
    
    # Returns an int corresponding to the relationship
    def relation_to_int(self, relation):
        
        if relation in self.relations_list:
            return self.relations_list.index(relation)
        
        else:
            return -1
        
        
    # Returns the relationship corresponding to an int
    def int_to_relation(self, index):
        
        if index >= 0 and index <= len(self.relations_list):
            return self.relations_list[index]
        
        else:
            if index == -1:
                return 'null'
            
            else:
                return 'invalid index'
        
        

    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Building the lists

Below is most of the code I used for building the above list
In general I took the most commonly occuring relationships of the firs 500K
found.  It's not perfect, but I think should be good for now.


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import string
from string import digits

import nltk, re

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


import pycorenlp, os
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('en')


relation_tuples = []

topic = "Georgetown University"
page = wiki_wiki.page(topic)

# Cycles through each sentence extracted from the page.
relation_tuple = []

for page in individual_pages:
    
    topic_page = wiki_wiki.page(page)
    sentences = prepContent(topic_page.text)
    
    for sentence in sentences:
        
        try:
    
            results = nlp.annotate(sentence, properties = {'annotators': 'openie', 'outputFormat': 'json'})
            sub_result = results['sentences'][0]['openie']
            
            for i in range(len(sub_result)):
                relation_tuple.append([sub_result[i]['subject'], sub_result[i]['relation'], sub_result[i]['object']])
                
        except:
            
            print("Something Happened.  Booooooooo!!!!")
            
my_list = []

for fun in relation_tuple:
    my_list.append(fun[1])
            
from collections import Counter
sorted = Counter(my_list).most_common(1000)

for i in relation_tuple:
    if 'is with' in i[1]:
        my_list.append(i)

"""