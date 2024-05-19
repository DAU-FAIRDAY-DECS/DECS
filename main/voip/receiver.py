import socket
import pyaudio

FORMAT = pyaudio.paInt16 # 16비트 정수 형식
CHANNELS = 1 # 단일 채널
RATE = 8000 # 전화 품질
CHUNK = 160 # 청크 크기
IP = "192.168.25.3" # 수신자 IP
PORT = 5005 # 포트 번호

def setup_audio_stream():
    audio = pyaudio.PyAudio()
    return audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

def create_udp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((IP, PORT))
    return sock

def main():
    stream = setup_audio_stream()
    sock = create_udp_socket()
    print("연결 대기 중...")
    data, addr = sock.recvfrom(CHUNK)
    if data.decode() == 'SYN':
        sock.sendto(b'ACK', addr)
        print("연결 확립")
    
    try:
        while True:
            data, _ = sock.recvfrom(320)
            stream.write(data)
    except KeyboardInterrupt:
        print("수신 종료")
    finally:
        stream.stop_stream()
        stream.close()
        sock.close()

if __name__ == '__main__':
    main()