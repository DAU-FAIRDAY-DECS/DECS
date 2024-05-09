import socket
import pyaudio
import sys

# 오디오 스트림 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

audio = pyaudio.PyAudio()

# 오디오 스트림 열기 (스피커 출력)
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, output=True,
                    frames_per_buffer=CHUNK)

# 네트워크 설정
ip = "0.0.0.0"
port = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port))

# 초기 'SYN' 메시지 대기
print("연결 대기 중...")
data, addr = sock.recvfrom(1024)
if data.decode() == 'SYN':
    sock.sendto(b'ACK', addr)
    print("연결 확립")

# 이후 데이터 수신 및 재생
while True:
    try:
        data, _ = sock.recvfrom(CHUNK)
        stream.write(data)
    except KeyboardInterrupt:
        print("수신 종료")
        break

# 자원 해제
stream.stop_stream()
stream.close()
audio.terminate()
sock.close()
