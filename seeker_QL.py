#!/usr/bin/env python3
# coding: utf-8

# This notebook implements a Q-learning algorithm for the seeker jetbot

import socket
import sys
import time
import random
import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
import imutils
import errno
import os
from datetime import datetime
from jetbot import Camera
from jetbot import Robot
from jetbot import bgr8_to_jpeg


# Global variables ======================================================
socket_seeker = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP/IP socket
ping = b'1' # arbitrary small data packet
robot = Robot() # seeker jetbot
camera_x = 224 # camera feed width
camera_y = 224 # camera feed height
collision_model = torchvision.models.alexnet(pretrained=False) # for collisions
device = torch.device('cuda')
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
normalize = torchvision.transforms.Normalize(mean, stdev)
camera = Camera.instance(width=camera_x, height=camera_y) # hider camera


# Functions =============================================================
# Action space
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

# Collision detection
def preprocess(camera_value):
	'''Transforms frame from the camera for posterior processing'''
	global device, normalize
	x = camera_value
	x = cv2.resize(x, (camera_x, camera_y))
	x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	x = x.transpose((2, 0, 1))
	x = torch.from_numpy(x).float()
	x = normalize(x)
	x = x.to(device)
	x = x[None, ...]
	return x

# Object detection
def crop_bottom_half(image):
	'''Returns bottom half of image'''
	cropped_img = image[int(image.shape[0]/2):image.shape[0], 0:image.shape[1], 0:image.shape[2]]
	return cropped_img

def detect_jetbot(image):
	'''Detects another jetbot and computes its center'''
	image = crop_bottom_half(image)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_green = np.array([5, 20, 5])
	upper_green = np.array([100, 150, 100])
	mask = cv2.inRange(hsv, lower_green, upper_green)
	cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = camera_x
	for c in cnts:
		area = cv2.contourArea(c)
		if (20000 > area > 200):
			x, y, w, h = cv2.boundingRect(c)
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
			M = cv2.moments(c)
			cx = int(M["m10"] / M["m00"])
			center = cx
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
	state_list[0] = detect_jetbot(image)
	
	# Set detection status
	if state_list[0] < (camera_x + 1):
		state_list[1] = 1
		message = '1'
	else:
		state_list[1] = 0
	request_detection = socket_seeker.recv(8)
	socket_seeker.sendall(message.encode())
	
	# Execute collision model to determine if blocked
	collision_output = collision_model(preprocess(image)).detach().cpu()
	prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])
	if prob_blocked > 0.7:
		state_list[2] = 1
	else:
		state_list[2] = 0
	
	# Encode state
	state = state_list[0] + (state_list[1]*2**0 + state_list[2]*2**1)*(camera_x + 1)
	
	return state_list, state

# Reward function
def compute_reward(action, state_list):
	'''Calculate the reward obtained by taking action from state_list'''
	# Reward seeing hider
	if state_list[1] == 1:
		reward = 1
	else:
		reward = -1
	
	# Penalise collisions
	if state_list[2] == 1 and action == 1:
		reward = -10

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


if __name__ == "__main__":
	# Connect to server
	server_address = ('192.168.0.21', 10000) # Check before every run
	socket_seeker.connect(server_address)
	action_space = [stop, forward, left, right]
	
	# Environment
	n_actions = len(action_space)
	n_states = (camera_x + 1)*2*2
	
	# Collision detection
	collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
	collision_model.load_state_dict(torch.load('best_model.pth'))
	collision_model = collision_model.to(device)

	# Q-learning ===========================================================
	# Q-table
	q_table = np.zeros((n_states, n_actions))
	
	# Hyperparameters
	n_train_episodes = 15     # number of rollouts for training
	max_steps = 100           # maximum number of steps per rollout

	gamma = 0.9               # discout factor
	alpha = 0.6               # learning rate

	epsilon = 1.0             # exploration-exploitation trade-off
	max_epsilon = 1.0
	min_epsilon = 0.01
	decay_rate = 0.01
	
	# Training stats
	rewards_training = np.zeros(n_train_episodes)
	timing_training = np.zeros(n_train_episodes)

	# Wait for user to toggle training
	socket_seeker.sendall(('SEEKER ready').encode())
	data_begin = socket_seeker.recv(8)

	for episode in range(n_train_episodes):
		# Reset stats
		episode_start = time.time()
		reward_episode = 0

		# Reset the environment
		state = update_state()[1]
		message = '0'
		print('Episode ' + str(episode))

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

		# Reduce epsilon
		epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

		# Log training stats
		rewards_training[episode] = reward_episode
		episode_stop = time.time()
		timing_training[episode] = episode_stop - episode_start

		# Toggle new episode
		socket_seeker.sendall(ping)
		data = socket_seeker.recv(8)

	# Save results and close ===============================================
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