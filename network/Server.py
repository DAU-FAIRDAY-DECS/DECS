import socket       

s = socket.socket()
host = socket.gethostname()
port = 12345
s.bind((host, port))
f = open('2.wav','wb')
s.listen(5)
while True:
    c, addr = s.accept()
    print ('Got connection from', addr)
    print ('Receiving...')
    l = c.recv(8096)
    while (l):
        print ('Receiving...')
        f.write(l)
        l = c.recv(8096)
    f.close()
    print ('Done Receiving')
    c.send('Thank you for connecting')
    c.close()