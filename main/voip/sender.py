import socket
import pyaudio

FORMAT = pyaudio.paInt16 # 16비트 정수 형식
CHANNELS = 1 # 단일 채널
RATE = 8000 # 전화 품질
CHUNK = 160 # 청크 크기
IP = "192.168.25.3" # 송신자 IP
PORT = 5005 # 포트 번호

def setup_audio_stream():
    audio = pyaudio.PyAudio()
    return audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def create_udp_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return sock

def main():
    stream = setup_audio_stream()
    sock = create_udp_socket()
    sock.sendto(b'SYN', (IP, PORT))
    try:
        data, addr = sock.recvfrom(CHUNK)
        if data.decode() == 'ACK':
            print("연결 확인 완료")
            while True:
                data = stream.read(CHUNK)
                sock.sendto(data, (IP, PORT))
        else:
            print("유효하지 않은 응답")
    except socket.timeout:
        print("응답 시간 초과")
    except KeyboardInterrupt:
        print("송신 종료")
    finally:
        stream.stop_stream()
        stream.close()
        sock.close()

if __name__ == '__main__':
    main()