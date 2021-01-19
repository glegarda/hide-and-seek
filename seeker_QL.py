#!/usr/bin/env python
# coding: utf-8

# This notebook implements a Q-learning algorithm for the seeker jetbot

##########################
#                        #
#  Import dependencies   #
#                        #
##########################

import socket
import sys

import time
import random

import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np

from jetbot import ObjectDetector
from jetbot import Camera
from jetbot import Robot
from jetbot import bgr8_to_jpeg

import errno
import os
from datetime import datetime


##########################
#                        #
#     Set up client      #
#                        #
##########################

# Create TCP/IP socket
socket_seeker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('62.174.227.135', 10000) # Check before every run
socket_seeker.connect(server_address)


##########################
#                        #
#  Create environment    #
#                        #
##########################

# Initialise seeker robot
robot = Robot()


# Define action space
def stop():
	'''Do not move'''
    robot.stop()
    
def forward():
	'''Take a step forward'''
    robot.forward(0.1)
    time.sleep(0.5)
    robot.stop()

def left():
	'''Rotate anticlockwise by a fixed amout'''
    robot.left(0.1)
    time.sleep(0.5)
    robot.stop()

def right():
	'''Rotate clockwise by a fixed amout'''
    robot.right(0.1)
    time.sleep(0.5)
    robot.stop()

action_space = [stop, forward, left, right]
n_actions = len(action_space)

# Define sate space
# State: {x,s,f} = {horizontal position of detected jetbot, seen/not seen, free/blocked}
camera_width = 225
n_states = (camera_width + 1)*2*2


# Collision detection
collision_model = torchvision.models.alexnet(pretrained=False)
collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
# SET CORRECT PATH IN FOLLOWING LINE ========================================================
collision_model.load_state_dict(torch.load('best_model.pth'))
# ===========================================================================================
device = torch.device('cuda')
collision_model = collision_model.to(device)

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
	'''Transforms frame from the camera for posterior processing'''
    global device, normalize
    x = camera_value
    x = cv2.resize(x, (224, 224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x


# Object detection
# PUT IN OUR ENGINE IN NEXT LINE ===========================================================
model = ObjectDetector('ssd_mobilenet_v2_coco.engine')
# ==========================================================================================
camera = Camera.instance(width=300, height=300)

def detection_center(detection):
    '''Computes the center x, y coordinates of the object'''
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return (center_x, center_y)

def detect_jetbot(image):
    '''Detects another jetbot and computes its center'''
    detections = model(image)
    # SET CORRECT LABEL VALUE IN NEXT LINE ==================================================
    matching_detections = [d for d in detections[0] if d['label'] == int(label_widget.value)]
    # =======================================================================================
    det = matching_detections[0]
    center = None
    if det is not None:
        center = detection_center(det)
    return center


# State observation
def update_state(): 
    '''Computes the 3 state components from received data'''
    # State with 3 components:
    #  - Horizontal position of detected hider jetbot
    #  - Seeing/not seeing hider
    #  - Free/blocked
    state_list = [None] * 3
    message = '0'
    
    # Detect hider jetbot
    image = camera.value
    center = detect_jetbot(image)
    
    # Set hider position and detection status
    if center is not None:
        state_list[0] = round(center[0])
        state_list[1] = 1
        message = '1'
    else:
        state_list[0] = None
        state_list[1] = 0
    socket_seeker.sendall(message.encode())
    
    # Execute collision model to determine if blocked
    collision_output = collision_model(preprocess(image)).detach().cpu()
    prob_blocked = float(F.softmax(collision_output.flatten(), dim=1)[0])
    if prob_blocked > 0.5:
        state_list[2] = 1
    else:
        state_list[2] = 0
    
    # Encode state
    state = state_list[0] + (state_list[1]*2**0 + state_list[2]*2**1)*301
    
    return state_list, state


# Define reward function
def compute_reward(action, state_list):
	'''Calculate the reward obtained by taking action from state_list'''
    # Reward seeing hider
    if state_list[1] == 1:
        reward = 1
    else:
        reward = -1
    
    # Penalise collisions
    if state_list[2] == 1 && action == 1:
        reward = -50

    return reward


# Define environment
def step(action):
    '''Executes an action, observes the state and computes the reward'''
    # Execute action
    action_space[action]()
    
    # Observe state
    state_list, state = update_state()
    
    # Compute reward
    reward = compute_reward(action, state_list)
    
    return state, reward


##########################
#                        #
#   Initialise Q-table   #
#                        #
##########################

q_table = np.zeros((n_states, n_actions))


##########################
#                        #
#  Set hyperparameters   #
#                        #
##########################

n_train_episodes = 100     # number of rollouts for training
max_steps = 240            # maximum number of steps per rollout

gamma = 0.9                # discount factor
alpha = 0.6                # learning rate

epsilon = 1.0              # exploration-exploitation trade-off
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01


##########################
#                        #
#         Train          #
#                        #
##########################

# Training stats
rewards_training = np.zeros(n_train_episodes)
timing_training = np.zeros(n_train_episodes)

# Wait for user to toggle training
socket_seeker.sendall('SEEKER ready').encode()
data = socket_seeker.recv(8)

for episode in range(n_train_episodes):
    # Reset stats
    episode_start = time.time()
    reward_episode = 0
    
    # Reset the environment
    state = update_state()
    step = 0
    message = '0'

    for iteration in range(max_steps):
        # Choose action according to exploration-exploitation tradeoff
        tradeoff = random.uniform(0, 1)
        if tradeoff > epsilon: # exploitation (on-policy)
            action = np.argmax(q_table[state, :])
        else: # exploration (off-policy)
            action = np.random.randint(n_actions)

        # Take action and observe new state and reward
        new_state, reward = step(action)
        reward_episode += reward

        # Update Q-table
        q_table[state, action] = q_table[state, action] + alpha*(reward + gamma*np.amax(q_table[new_state, :])
                                                                 - q_table[state, action])

        # Update state
        state = new_state
        
        # Send status to server
        if iteration == max_steps:
            message = '1'
        socket_seeker.sendall(message.encode())
    
    # Reduce epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
    # Log training stats
    rewards_training[episode] = reward_episode
    episode_stop = time.time()
    timing_training[episode] = episode_stop - episode_start
    
    # Toggle new episode
    data = socket_seeker.recv(8)


##########################
#                        #
# Save results and close #
#                        #
##########################

# Create timestamped directory to save data
new_dir = os.path.join(os.getcwd(), datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
try:
    os.makedirs(new_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise  # This was not a "directory exist" error.

# Save data
np.savetxt(new_dir+'/Q_table_S', q_table, delimiter=',')
np.savetxt(new_dir+'/Rewards_training_S', rewards_training, delimiter=',')
np.savetxt(new_dir+'/Timing_training_S', timing_training, delimiter=',')

# Close
camera.stop()
socket_seeker.close()

