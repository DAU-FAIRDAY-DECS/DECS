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
IP = "0.0.0.0"
PORT = 5005

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
            data, _ = sock.recvfrom(1024)
            if len(data) < 4:
                continue
            seq_num = int.from_bytes(data[:4], 'big')
            compressed_data = data[4:]
            decompressed_data = zlib.decompress(compressed_data)
            stream.write(decompressed_data)
    except KeyboardInterrupt:
        print("수신 종료")
    finally:
        stream.stop_stream()
        stream.close()
        sock.close()

if __name__ == '__main__':
    main()