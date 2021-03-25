# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Python Libraries
import string
from string import digits
import tensorflow as tf
import nltk
import numpy as np
from pycorenlp import StanfordCoreNLP
import re
import sys
import random
import pickle
import networkx as nx
import wikipediaapi
import operator
import math
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sklearn
from sklearn.metrics import classification_report


# Self-Made Files
import classes
import relation_standardizer

# import node
# nltk.download('punkt')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Function Defs

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""
Prepare the content of a text passage
 - Separate into sentences
 - Lower case, remove quotes, remove punctuation, remove extra spaces
"""""""""""""""

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
    

"""""""""""""""
Loads pre-processed Georgetown data
"""""""""""""""

def load_GU_Data():

    # Location of Scipt being run
    abspath = sys.path[0]
    
    # Loading datasets previously constructed
    G_augment =         pickle.load(open(abspath + "/Other-Data/G_augmented.pkl", "rb"))
    cleaned_tuples =    pickle.load(open(abspath + "/Other-Data/cleaned_tuples.pkl", "rb"))
    topic_graph =       pickle.load(open(abspath + "/Other-Data/Topic_Graph.pkl", "rb"))
    encoded_tuples =    pickle.load(open(abspath + "/Other-Data/relation_tuples.pkl", "rb"))
    
    return G_augment, cleaned_tuples, topic_graph, encoded_tuples


# I got this somewhere on Wiki and can't find the page again.  When I can I'll
# expand this so a new topic can automatically be downloaded.
# https://en.wikipedia.org/wiki/Special:Export
    
def GU_pages():
    
    return [
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



# Recursive method to retrieve all the pages in a category page of wikipedia
# inlcusive of nested category pages.

def get_all_children_pages(categories, wiki_wiki):
    good_list = []
    input_dict = categories.categorymembers
    keys = input_dict.keys()
    
    for key in keys:
        
        if "Category" in key:
            
            # I'm not taking the alumni pages specifically to reduce the #
            # of pages.  This reduced it slightly more than 1/2.
            if "alumni" not in key:
                temp = get_all_children_pages(wiki_wiki.page(key), wiki_wiki)
                good_list += temp
            
        else:
            
            good_list.append(key)
            
    return good_list


def get_pages(Page_List, wiki_wiki):
    individual_pages = []
    Topic_Dict = {}

    # Builds the list
    for item in Page_List:
        if "Category" in item:
            output_list = get_all_children_pages(wiki_wiki.page(item), wiki_wiki)
            individual_pages += output_list
            
        else:
            individual_pages.append(item)
    
    # Puts the pages into a dictionary for easy access.
    for page in individual_pages:
        Topic_Dict[page] = wiki_wiki.page(page)

    return individual_pages, Topic_Dict

#G = G_augment

#G_augment[topic]

def determine_node_knowledge(current_node, G, sess, critic):
    
    connecting_nodes = list(G.adj[current_node])

    # A single name of a node
    # node = connecting_nodes[0]
    node_info = []

    for node in connecting_nodes:
        
        encodes = G.nodes[node]['encodes']
        
        pred_rewards = sess.run(critic.layer3, feed_dict = {critic.observation: np.reshape(encodes,(-1, 768))})
        
        node_info.append([node, np.mean(pred_rewards), np.std(pred_rewards)])
        
    return node_info

def explore_new_page(current_node, bc, nlp, wiki_wiki, G, rs):
    
    topic_page = wiki_wiki.page(current_node)
    sentences = prepContent(topic_page.text)
    
    _relation_tuples = []
    _cleaned_tuples = []
    _encodes = []
    
    print('Currently processing page: ' + current_node)
    
    for sentence in sentences:
        
        try:
    
            results = nlp.annotate(sentence, properties = {'annotators': 'openie', 'outputFormat': 'json'})
            sub_result = results['sentences'][0]['openie']
            
            for i in range(len(sub_result)):
                _relation_tuples.append([sub_result[i]['subject'], sub_result[i]['relation'], sub_result[i]['object']])
                
        except:
            
            print("Something Happened.  Booooooooo!!!!")


    for tup in _relation_tuples:
        relation = rs.standardize(tup[1])
        
        if relation != 'null':
            _cleaned_tuples.append([tup[0], relation, tup[2]])

    # Need to add code to create encodes from BERT
    
    for relation in _cleaned_tuples:

        _encodes.append([relation[1], bc.encode([relation[0] + ' ||| ' + relation[2]])]) 
        
    # Need to add node to G along with summary, relation_tuples, and encodes
    
    G_relations = []
    G_encodes = []
    
    node = G.nodes[current_node]
    
    try:
        sentences = prepContent(node['summary'])
        
        for sentence in sentences:
            try:
                results = nlp.annotate(sentence, properties = {'annotators': 'openie', 'outputFormat': 'json'})
                sub_result = results['sentences'][0]['openie']
                for i in range(len(sub_result)):
                    G_relations.append([sub_result[i]['subject'], sub_result[i]['relation'], sub_result[i]['object']])
            except: 
                print("Problem in Core-NLP.  Skipping")
            
        if len(G_relations) > 10:
            G_relations = random.sample(G_relations, 10)
            
        for relation in G_relations:
            try:
                G_encodes.append(bc.encode([relation[0] + " ||| " + relation[2]]))
            except:
                print("Problem in bc encoding.  Skipping")
    except:
        print("Problem in Summary.  Leaving empty.")
        
            
    node['relations'] = G_relations
    node['encodes'] = G_encodes
        
    return _cleaned_tuples, _encodes
    
    
def train_on_topic(buffer, actor_tgt, actor_av , critic, sess, rs, encoded_tuples):
    
    transfer_weights = [tf.assign(tgt, av) for (tgt, av) in zip(tf.trainable_variables('target'), tf.trainable_variables('value'))]
    
    num_guesses = 0
    guesses = []    
    
    for episode in range(5000):
        
        if episode % 50 == 0:
            print("Currently at Episode " + str(episode) + " of 5000")
            
        if episode % 20 == 0:
            sess.run(transfer_weights)
            
        relations = []
        probs = []
        chosen = []
        rewards = []
        states = []
        pred_rewards = []

        # Run the Training Routine for the Critic
        training_sample = buffer.sample(100)
        
        # sample = training_sample[0]
        
        # Use the Actor to determine the predicted relation for a state
        for sample in training_sample:
            relations.append(rs.relation_to_int(sample[0]))
            states.append(sample[1])
            probs.append(sess.run(actor_tgt.layer3, feed_dict = {actor_tgt.observation: sample[1]}))

        # Formatting the probabilities to make them easier to use
        for prob in probs:
            prob_list = prob[0].tolist()
            chosen.append(prob_list.index(max(prob_list)))

        # Determine reward from the environment       
        for actual, pred in zip(relations, chosen):
            if actual == pred:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        # Training the Critic
        critic_loss, _ = sess.run([critic.loss, critic.train], feed_dict = {critic.observation: np.reshape(states,(-1, 768)), critic.reward: np.reshape(rewards, (-1,1))})

        # Predict the rewards for the new relations from the page.  
        # Runs it through the newly trained critic and then flattens it.
        pred_rewards = sess.run(critic.layer3, feed_dict = {critic.observation: np.reshape(states,(-1, 768))})
        pred_rewards = [item for sublist in pred_rewards for item in sublist]
      
        # td_error = [(r - p) for p, r in zip(pred_rewards, rewards)] 
       
        # Train the Actor
        for s, r, p in zip(states, relations, pred_rewards):
                        
            actor_prob = sess.run(actor_tgt.layer3, feed_dict = {actor_tgt.observation: s})
            # print (actor_prob)
            actor_prob = actor_prob[0].tolist() 
            chosen = actor_prob.index(max(actor_prob))
            num_guesses += 1
            
            # I need to come back and make sure this is correct
            reward = -p
            if chosen == r:
                reward = 1 + reward
                guesses.append(1)
            else:
                guesses.append(0)

            actor_loss, _ = sess.run([actor_av.loss, actor_av.train], feed_dict = {actor_av.observation: s, actor_av.td_error: reward, actor_av.relation: r})
            # print(guesses[-20:])
            
        # print(questions)
        if episode % 50 == 0:
            print ("Training loss for critic is: " + str(critic_loss) + "   Training loss for actor is: " + str(actor_loss))
        if num_guesses >= 500 and episode % 20 ==0:
            print ("Accuracy over last 500 guesses: " + "{:.2%}".format(sum(guesses[-500:]) / 500))
            
        
    # Graph for Training on Georgetown Data    
    means = []
    for i in range(300, len(guesses)):
        means.append(np.mean(guesses[i - 300: i - 1]))


    plt.plot(means)
    plt.ylabel('Average Score (Over 200 Guesses)')
    plt.xlabel('Guesses')
    plt.title("Initial Agent Training on Georgetown Dataset")
    # plt.savefig(str(env) + '.png')
    plt.savefig('Initial_Training.png')
        
def save_session(name, sess):

    abspath = sys.path[0]

    saver = tf.train.Saver()
    save_path = saver.save(sess, abspath + "/Other-Data/" + name + ".ckpt")
    print("Model saved in path: %s" % save_path)
    
def load_session(name, sess):
    
    abspath = sys.path[0]

    saver = tf.train.Saver()
    saver.restore(sess, abspath + "/Other-Data/" + name + "_trained.ckpt")
    print("Model restored.")
    
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Main Program

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def run(args):

    """""""""""""""""""""""""""""""""
    
    Set up
    
    """""""""""""""""""""""""""""""""
    
    # topic = "Georgetown_University"
    topic =         args.topic
    actor_lr =      args.actor_LR
    critic_lr =     args.critic_LR
    episodes =      args.eps
    load_sess =     args.load_sess
    sess_name =     args.sess_name
    
    topic =         "Georgetown_University"
    actor_lr =      0.0001
    critic_lr =     0.0005
    episodes =      227
    
    buffer = classes.ReplayBuffer()
    wiki_wiki = wikipediaapi.Wikipedia('en')
    rs = relation_standardizer.Relation_Standardizer("Bob")
    
    # TODO: Future add code to let a person choose another topic

    G_augment, cleaned_tuples, topic_graph, encoded_tuples = load_GU_Data()
    page_list = GU_pages()
    buffer.relations = encoded_tuples
    individual_pages, Topic_Dict = get_pages(page_list, wiki_wiki)
        
    n_features =    768     # Default used in BERT
    n_output =      25      # Num possilbe relations.  Currently set to 25
    actor_H1 =      200     # Num of hidden units in first layer of actor
    actor_H2 =      200     # Num of hidden units in second layer of actor
    critic_H1 =     200     # Num of hidden units in first layer of critic
    critic_H2 =     200     # Num of hidden units in second layer of critic
    
    # TensorFlow Setup and Initialization
    tf.reset_default_graph()    
    
    actor_tgt = classes.Actor(n_features, n_output, actor_lr, actor_H1, actor_H2, "target")
    actor_av = classes.Actor(n_features, n_output, actor_lr, actor_H1, actor_H2, "value")
    transfer_weights = [tf.assign(tgt, av) for (tgt, av) in zip(tf.trainable_variables('target'), tf.trainable_variables('value'))]

    critic = classes.Critic(n_features, critic_lr, critic_H1, critic_H2)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # BERT Setup
    print("In a terminal window Python Environment can access, run the following:\n" + "bert-serving-start -model_dir ~/Final_Proj/BERT-Data/ -num_worker=2\n\nPress Enter when done.")
    x = input()
    
    from bert_serving.client import BertClient
    bc = BertClient()
    
    # Core-NLP Setup
    print("In a terminal window run the following:\ncd ~/Final_Proj/Stan-Core-NLP; java -mx6g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000\n\nPress Enter when done.")
    x = input()
    
    nlp = StanfordCoreNLP('http://localhost:9000')
    
    current_node = topic
    
    #Note: Ability to train on other topics is not currently implemented.
    if (load_sess):
        load_session(sess_name, sess)
    else:
        train_on_topic(buffer, actor_tgt, actor_av, critic, sess, rs, encoded_tuples)
    
    # Setup for performance tracking
    guesses = []
    relation_accuracy = []
    questions = []
    transfer_index = 0
    actor_loss = 0
    
    
    """""""""""""""""""""""""""""""""
    
    Running Episodes
    
    """""""""""""""""""""""""""""""""

    for episode in range(episodes):
        
        # Code to transfer weights every ~1K guesses
        if math.floor(len(guesses)/1000) > transfer_index:
            sess.run(transfer_weights)
            transfer_index += 1
            print("Weights Transfered")
        
        # Code to keep track of stuff during episode
        relations = []
        probs = []
        chosen = []
        rewards = []
        states = []
        pred_rewards = []
        
        # Run the Training Routine for the Critic
        training_sample = buffer.sample(100)
       
        # Use the Actor to determine the predicted relation for a state
        for sample in training_sample:
            relations.append(rs.relation_to_int(sample[0]))
            states.append(sample[1])
            probs.append(sess.run(actor_tgt.layer3, feed_dict = {actor_tgt.observation: sample[1]}))

        # Formatting the probabilities to make them easier to use
        for prob in probs:
            prob_list = prob[0].tolist()
            chosen.append(prob_list.index(max(prob_list)))

        # Determine reward from the environment       
        for actual, pred in zip(relations, chosen):
            if actual == pred:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                
        # Training the Critic
        critic_loss, _ = sess.run([critic.loss, critic.train], feed_dict = {critic.observation: np.reshape(states,(-1, 768)), critic.reward: np.reshape(rewards, (-1,1))})
        if episode > 1:
            print("Training loss for critic is: " + str(critic_loss) + "  Training loss for actor is: " + str(actor_loss))
        
        ######
        # Exploration code
        ######
        
        # Run the links available for the current node through the critic to get lowest mean
        # The std is also included if we want to include a LCB version later.
        node_predictions = determine_node_knowledge(current_node, G_augment, sess, critic)
        
        # Filter out nan entries & sort
        filtered = [x for x in node_predictions if not math.isnan(x[1])]
        filtered.sort(key = lambda x: x[1])

        # Determine the next node to go to
        for node in filtered:
            if node[0] not in individual_pages:
                current_node = node[0]
                individual_pages.append(current_node)
                
                break
        
        # Explore page 
        clean_tuples, encodes = explore_new_page(current_node, bc, nlp, wiki_wiki, G_augment, rs)
        
        # Add encoded tuples to the replay buffer
        buffer.relations += encodes
        
        # To hold the information for training the Actor
        relations = []
        states = []
        chosen = []
        probs = []

        # Drawing a new random sample from the buffer.
        # This trains the agen 1/2 on new data and 1/2 on old data        
        from_buffer = buffer.sample(len(encodes))
        encodes += from_buffer
        
        # Gather info for training the Actor
        for encode in encodes:
            relations.append(rs.relation_to_int(encode[0]))
            states.append(encode[1])
            probs.append(sess.run(actor_tgt.layer3, feed_dict = {actor_tgt.observation: encode[1]}))

        # Predict the rewards for the new relations from the page.  
        # Runs it through the critic and then flattens it.
        pred_rewards = sess.run(critic.layer3, feed_dict = {critic.observation: np.reshape(states,(-1, 768))})
        pred_rewards = [item for sublist in pred_rewards for item in sublist]

        # Train the Actor on the downloaded items.
        for s, r, p, clean in zip(states, relations, pred_rewards, clean_tuples):

            actor_prob = sess.run(actor_tgt.layer3, feed_dict = {actor_tgt.observation: s})
            actor_prob = actor_prob[0].tolist() 
            chosen = actor_prob.index(max(actor_prob))
            
             # Record the actual relation extracted from data and chosen
             # Use it later to build a table for accuracy by relation
            relation_accuracy.append([r, chosen])           
            
            """""""""""""""""""""""""""""""""""""""""
            Calculation of TD error.
            
            Formula for TD error is normally:
                R + gamma * (v_t+1) - v(t)
                
            We can't really guess the next state with any accuracy so we're 
            using a Contextual Bandit Actor Critic model.  This means the 
            TD error we're going to use is based on:
                R - v(t)
                
                where R = 1 turn reward and v(t) is a one turn expected value.
            
            Because our reward structure = 1 if correct 0, otherwise, we're
            settup up the -v(t) first, and then adding 1 if guess is correct.
            """""""""""""""""""""""""""""""""""""""""
            reward = -p
            
            if chosen == r:
                reward = 1 + reward
                guesses.append(1)
            else:
                questions.append("Actual:    " + clean[0] + " | " + clean[1] + " | " + clean[2] + "\nPredicted: " + clean[0] + " | " + str(rs.int_to_relation(chosen)) + " | " + clean[2])
                guesses.append(0)
                
            actor_loss, log_prob,  _ = sess.run([actor_av.loss, actor_av.log_probability, actor_av.train], feed_dict = {actor_av.observation: s, actor_av.td_error: reward, actor_av.relation: r})
            # print(log_prob)
            
        
            
        # print(questions)
        print ("Have explored " + str(episode + 1) + " new pages and made " + str(len(guesses)) + " guesses.  Accuracy (last 200 predictions) was: " + "{:.2%}".format(sum(guesses[-200:]) / min(200, len(guesses))))
    
    save_session(sess_name, sess)
    
    """""""""""""""""""""""""""""""""
    Performance Graphing
    
    """""""""""""""""""""""""""""""""

    # Graph for Training on Georgetown Data    
    means = []
    for i in range(3000, len(guesses), 3000):
        means.append(np.mean(guesses[i - 3000: i - 1]))
        
    ticks = list(range(3000, len(guesses), 3000))

    plt.plot(ticks, means)
    plt.ylabel('Average Score (Over 3000 Guesses)')
    plt.xlabel('Guesses')
    plt.title("Agent Training on New Pages")
    plt.savefig('Accuracy_New_Pages_2.png')
    
    # List of Guesses by Actual Relation
    
    first = list(range(25))
    
    y_true, y_pred = [], []
    
    for i in relation_accuracy:
        y_true.append(i[0])
        y_pred.append(i[1])
    

    target_relations = []
    for i in first:
        target_relations.append(rs.int_to_relation(i))
        
    print(classification_report(y_true, y_pred, target_names = target_relations, labels = first))
    
    
    y_true, y_pred = [], []
    
    for i in relation_accuracy[-50000:]:
        y_true.append(i[0])
        y_pred.append(i[1])
    

    target_relations = []
    for i in first:
        target_relations.append(rs.int_to_relation(i))
        
    print(classification_report(y_true, y_pred, target_names = target_relations, labels = first))


"""""""""""""""""""""""""""""""""
Argument Parsing

"""""""""""""""""""""""""""""""""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required = True)
    parser.add_argument("-topic", type = str, default = "Georgetown_University")
    parser.add_argument("-actor_LR", type = float, default = 0.0001)
    parser.add_argument("-critic_LR", type = float, default = 0.0005)
    parser.add_argument("-eps", type = int, default = 2000)
    parser.add_argument("-load_sess", type = bool, default = True)
    parser.add_argument("-sess-name", type = str, default = "model_initial_training")


    #TODO: add your own parameters

    args = parser.parse_args()
    run(args)

