import socketio
import time 
# Create a Socket.IO client
sio = socketio.Client()

# # Event handler for connection
@sio.event
def connect():
    print('Connected to the server')
    sio.emit('new-connection', 'drive-clinet')

    

# Event handler for disconnection
@sio.event
def disconnect():
    print('Disconnected from the server')

# Connect to the server
sio.connect('http://localhost:3000')

# Wait for the connection to be established before sending more messages
i = 1
while True:
    sio.emit('drive-client', str(i))
    i+=1
    time.sleep(1) # import time

sio.wait()