# Wikipedia Relationship Predictor

This repository contains the beginning code for a reinforcement learning agent that is designed to navigate links on Wikipedia, build a graph network of the links to other topics, and learn to predict relationships between nodes.  

## General Algorithm

The AI agent itself is a Contextual Bandits Actor-Critic model (Sutton, 2018) implementing several features from the DQN model (Mnih, 2015) to improve stability, notably the replay buffer.

## Current Progress

The code is still messey and not yet very well modularized, but I hope to work on that in the future.  The agent itself showed promise and with some additional fine-tuning, I believe could be extended to many more relationships.  The main program is contained in BaselineQG_v5.py with some helper functions in the additional files.