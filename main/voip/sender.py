import pyaudio
import socket
import zlib
import threading
import wave
import random
import time

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000

PORT = 9001
SENDER_CONTROL_PORT = 9002  # 송신자 제어 메시지 포트
RECEIVER_CONTROL_PORT = 9003  # 수신자 제어 메시지 포트
RECEIVER_IP = '10.10.1.90'  # 수신자 IP 주소

def send_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_sock.bind(('0.0.0.0', SENDER_CONTROL_PORT))
    server_address = (RECEIVER_IP, PORT)
    
    print("송신 대기")
    
    wf = wave.open("input.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    def check_for_control_messages():
        while True:
            msg, _ = control_sock.recvfrom(1024)
            if msg.decode() == "END":
                break
        control_sock.close()
        print("송신 연결 해제")
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()
        wf.close()
        exit(0)

    control_thread = threading.Thread(target=check_for_control_messages)
    control_thread.start()

    first_packet = True

    try:
        while True:
            data = stream.read(CHUNK)
            wf.writeframes(data)
            compressed_data = zlib.compress(data)
            # 패킷 손실 시뮬레이션 (10% 확률로 패킷 손실)
            if random.random() > 0.1:
                # 패킷 지연 시뮬레이션 (0~100ms 랜덤 지연)
                time.sleep(random.uniform(0, 0.1))
                sock.sendto(compressed_data, server_address)
            else:
                # 송신된 패킷 손실 시 빈 패킷을 보냄
                sock.sendto(zlib.compress(bytes([0]*CHUNK)), server_address)
            if first_packet:
                control_sock.sendto(b"START", (RECEIVER_IP, RECEIVER_CONTROL_PORT))
                first_packet = False
    except KeyboardInterrupt:
        control_sock.sendto(b"END", (RECEIVER_IP, RECEIVER_CONTROL_PORT))
        print("송신 연결 해제")
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()
        control_sock.close()
        wf.close()

if __name__ == "__main__":
    send_audio()
