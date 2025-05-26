import socketio
import time 
# Create a Socket.IO client
sio = socketio.Client()

# # Event handler for connection
@sio.event
def connect():
    print('Connected to the server')

def callback(msg):
    print('killing all processes')




sio.on('kill', callback)
    

# Event handler for disconnection
@sio.event
def disconnect():
    print('Disconnected from the server')

# Connect to the server
sio.connect('http://localhost:3000')



sio.wait()