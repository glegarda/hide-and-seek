#!/usr/bin/env python3
# coding: utf-8

# This notebook defines a server class that handles the communication with the jetbots. 
# It has two duties: to toggle the beginning of a new episode and to signal the hider
# robot that it is in the field of view of the seeker robot.

import socket
import sys
import threading
import signal


def signal_handler(signal, frame):
	print('\nKeyboard interrupt! Shutting down server...')
	sock.shutdown(socket.SHUT_RDWR)
	

class Server:
	def __init__(self):
		self.clients = []             # List of connected clients: [seeker, hider]
		self.ip = 'localhost'         # Server IP address
		self.port = 10000             # Server port number
		self.buffer_size = 16         # Buffer size
		self.ping = b'1'              # Arbitrary small data packet

	def start_server(self):
		try:
			global sock
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			sock.bind(('', self.port))
			sock.listen(5)
			for i in range(2):
				client_socket, addr = sock.accept()
				self.clients.append(client_socket)
				if i == 0:
					print('Connected with ' + addr[0] + ':' + str(addr[1]) + '--> SEEKER')
				else:
					print('Connected with ' + addr[0] + ':' + str(addr[1]) + '--> HIDER')
			thread = threading.Thread(target=self.threaded_clients)
			thread.start()
			thread.join()
			sock.shutdown(socket.SHUT_RDWR)
			sock.close()
		except OSError as error:
			sock.close()
			sys.exit()
	

	def threaded_clients(self):
		# Wait until both hider and seeker are ready to begin training
		for client in self.clients:
			data_ready = client.recv(self.buffer_size).decode()
			print(data_ready)
		input('Press ENTER to begin training:')
		# Send signal to begin training
		for client in self.clients:
			client.sendall(self.ping)
		# Training loop
		for episodes in range(15):
			# Receive and process detection status
			request_detection = self.clients[1].recv(self.buffer_size)
			self.clients[0].sendall(request_detection)
			data_detection = self.clients[0].recv(self.buffer_size)
			self.clients[1].sendall(data_detection)
			for iterations in range(100):
				# Receive and process detection status
				request_detection = self.clients[1].recv(self.buffer_size)
				self.clients[0].sendall(request_detection)
				data_detection = self.clients[0].recv(self.buffer_size)
				self.clients[1].sendall(data_detection)
			# Toggle new episode
			for client in self.clients:
				data_episode = client.recv(self.buffer_size).decode()
			input('Episode done. Reset the initial configuration and press ENTER to begin a new episode:')
			for client in self.clients:
				client.sendall(self.ping)
		# Clean up
		print('Training completed. Shutting down server...')
		for client in self.clients:
			client.close()
		self.clients.clear() 

				
if __name__ == "__main__":
	# Handle keyboard interrupt (Ctrl-C)
	signal.signal(signal.SIGINT, signal_handler)

	# Start server
	server = Server()
	server_thread = threading.Thread(target=server.start_server)
	server_thread.start()
	server_thread.join()

