import socket

s = socket.socket()
host = socket.gethostname()
port = 12345

s.connect((host, port))
f = open('1.wav','rb')
print ('Sending...')
l = f.read(8096)
while (l):
    print ('Sending...')
    s.send(l)
    l = f.read(8096)
f.close()
print ('Done Sending')
s.shutdown(socket.SHUT_WR)
print (s.recv(8096))
s.close()