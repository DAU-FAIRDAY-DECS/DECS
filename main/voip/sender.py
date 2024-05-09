import socket
import pyaudio
import sys

# 오디오 스트림 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 256

audio = pyaudio.PyAudio()

# 오디오 스트림 열기 (마이크 입력)
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# 네트워크 설정
ip = "192.168.1.101"  # 수신자의 IP 주소로 변경
port = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 연결 확인을 위해 'SYN' 메시지 전송
sock.sendto(b'SYN', (ip, port))
try:
    # 응답 대기
    sock.settimeout(5)  # 5초 대기
    data, addr = sock.recvfrom(CHUNK)
    if data.decode() == 'ACK':
        print("연결 확인 완료")
    else:
        print("유효하지 않은 응답")
        sys.exit()
except socket.timeout:
    print("응답 시간 초과")
    sys.exit()

# 연결이 성공적으로 확인된 후 오디오 데이터 전송 시작
while True:
    try:
        data = stream.read(CHUNK)
        sock.sendto(data, (ip, port))
    except KeyboardInterrupt:
        print("송신 종료")
        break

# 자원 해제
stream.stop_stream()
stream.close()
audio.terminate()
sock.close()
