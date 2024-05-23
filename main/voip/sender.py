# zlib 모듈 활용
# LZ77 알고리즘
# 허프만 코딩

import socket
import pyaudio
import zlib

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
CHUNK = 160
IP = "192.168.0.13"
PORT = 5005

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
    seq_num = 0
    try:
        data, addr = sock.recvfrom(CHUNK)
        if data.decode() == 'ACK':
            print("연결 확인 완료")
            while True:
                data = stream.read(CHUNK)
                compressed_data = zlib.compress(data, level=9)
                packet = seq_num.to_bytes(4, 'big') + compressed_data
                sock.sendto(packet, (IP, PORT))
                seq_num += 1
                if seq_num > 2**32 - 1:
                    seq_num = 0
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