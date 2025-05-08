import socket

# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('0.0.0.0', 5005)  # Listen on all available interfaces
print(f'Starting up server on {server_address}')
server_socket.bind(server_address)

# Listen for incoming connections
server_socket.listen(1)

try:
    while True:
        # Wait for a connection
        print('Waiting for a connection...')
        connection, client_address = server_socket.accept()
        try:
            print(f'Connection from {client_address}')
            
            # Receive hand identifier
            data = connection.recv(16).decode('utf-8').strip()
            print(f'Received hand identifier: {data}')
            
            # Send start command
            print('Sending start command...')
            connection.sendall(b'S\n')
            
            # Receive IMU data
            count = 0
            while count < 100:  # Receive 100 data points
                data = connection.recv(128).decode('utf-8').strip()
                print(f'Received: {data}')
                count += 1
            
            # Send end command
            print('Sending end command...')
            connection.sendall(b'End\n')
            
        finally:
            # Clean up the connection
            connection.close()
            
except KeyboardInterrupt:
    print('Server is shutting down')
    server_socket.close()