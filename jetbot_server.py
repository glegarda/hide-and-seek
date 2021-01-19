#!/usr/bin/env python
# coding: utf-8

# This notebook defines a server class that handles the communication with the jetbots. 
# It has two duties: to toggle the beginning of a new episode and to signal the hider
# robot that it is in the field of view of the seeker robot.

import socket
import sys
import threading

class Server:
	def __init__(self):
		self.clients = []             # List of connected clients: [seeker, hider]
		self.ip = 'localhost'         # Server IP address
		self.port = 10000             # Server port number
		self.buffer_size = 16         # Buffer size
		self.hider_ready = False      # True if hider is ready to begin training
		self.seeker_ready = False     # True if seeker is ready to begin training
		self.begin_training = False   # True to toggle the beginning of the training process
		self.episode_ended = False    # True if the hider has finished the current episode
		self.detection_set = False    # True if seeker data has been processed
		self.detected = '0'           # Detection status (1: visual contact; 0: no visual contact)
		self.ping = '1'               # Arbitrary small data packet

	def start_server(self):
		try:
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock.bind((self.ip, self.port))
			sock.listen(5)
			while True:
				client_socket, addr = sock.accept()
				self.clients.append(client_socket)
				if len(self.clients) == 1:
					print ('Connected with ' + addr[0] + ':' + str(addr[1]) + '--> SEEKER')
					threading.Thread(target=self.threaded_seeker, args=(client_socket,)).start()
				elif len(self.clients) == 2:
					print ('Connected with ' + addr[0] + ':' + str(addr[1]) + '--> HIDER')
					threading.Thread(target=self.threaded_hider, args=(client_socket,)).start()
			sock.close()
		except socket.error as msg:
			print ('Could not start SERVER thread.')
			sys.exit()
	
	def threaded_seeker(self, client_socket):
		# Wait until both hider and seeker are ready to begin training
		data = client_socket.recv(self.buffer_size).decode()
		if data:
			print(data)
			self.seeker_ready = True
			while not self.hider_ready:
				pass
			input("Press ENTER to begin training:")
			self.begin_training = True
		# Training loop
		while self.begin_training:
			# Receive and process detection status
			data = client_socket.recv(self.buffer_size).decode()
			if data:
				self.detected = data
				self.detection_set = True
			else:
				break
			# Receive episode status and toggle the next one
			data = client_socket.recv(self.buffer_size).decode()
			if data:
				if data == '1':
					while not self.episode_ended:
						pass
					input("Episode done. Press ENTER to begin next episode:")
					for client in self.clients:
						client.send(ping.encode())
					self.episode_ended = False
			else:
				break
		self.clients.remove(client_socket)
		client_socket.close()
		
	def threaded_hider(self, client_socket):
		# Wait for hider to be ready to begin training
		data = client_socket.recv(self.buffer_size).decode()
		if data:
			print(data)
			self.hider_ready = True
		# Wait for user to toggle the beginning of the training process
		while not self.begin_training:
			pass
		# Training loop
		while self.begin_training:
			# Wait for seeker and send detection status
			while not self.detection_set:
				pass
			client_socket.send(self.detected.encode())
			self.detection_set = False
			# Receive and process episode status
			data = client_socket.recv(self.buffer_size).decode()
			if data:
				if data == '1':
					self.episode_ended = True
			else:
				break
		self.clients.remove(client_socket)
		client_socket.close()

# Start server
server = Server()
threading.Thread(target=server.start_server).start()

