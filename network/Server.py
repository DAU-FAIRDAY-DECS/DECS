import socket
import random

# 소켓 객체를 생성
s = socket.socket()
host = socket.gethostname()  
port = 12345                
s.bind((host, port))         
f = open('2.png', 'wb')   
s.listen(5)

# 받은 데이터와 손실된 데이터의 총 바이트 수
total_received_bytes = 0
total_lost_bytes = 0

while True:
    c, addr = s.accept() 
    print('Got connection from', addr)
    print('Receiving...')
    l = c.recv(8096)

    while l:
        # 10% 확률
        if random.random() > 0.1:
            print('Receiving...')
            f.write(l) 
            total_received_bytes += len(l)
        else:
            print('Packet loss occurred')
            total_lost_bytes += len(l)
        l = c.recv(8096)
    
    f.close()
    print('Done Receiving')
    print(f'Total received bytes: {total_received_bytes}')
    print(f'Total lost bytes: {total_lost_bytes}')
    
    # 손실률을 계산하고 출력
    if total_received_bytes + total_lost_bytes > 0:
        loss_percentage = (total_lost_bytes / (total_received_bytes + total_lost_bytes)) * 100
        print(f'Loss percentage: {loss_percentage:.2f}%')
    
    c.send(b'Thank you for connecting')
    c.close()
