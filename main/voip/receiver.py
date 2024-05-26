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
SENDER_IP = '10.10.1.90' # 송신자 IP 주소

def receive_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', PORT))
    control_sock.bind(('0.0.0.0', RECEIVER_CONTROL_PORT))
    
    print("수신 대기")
    
    wf = wave.open("output.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    def check_for_control_messages():
        while True:
            msg, _ = control_sock.recvfrom(1024)
            if msg.decode() == "END":
                break
        control_sock.close()
        print("수신 연결 해제")
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()
        wf.close()
        exit(0)
    
    control_thread = threading.Thread(target=check_for_control_messages)
    control_thread.start()

    try:
        compressed_data, addr = sock.recvfrom(2048)
        control_sock.sendto(b"START", (SENDER_IP, SENDER_CONTROL_PORT)) # 송신자에게 연결 완료 메시지 전송
        print("수신 연결 완료")
        while True:
            # 패킷 지연 시뮬레이션 (0~100ms 랜덤 지연)
            time.sleep(random.uniform(0, 0.1))
            data = zlib.decompress(compressed_data)
            wf.writeframes(data)
            stream.write(data)
            compressed_data, _ = sock.recvfrom(2048)
    except KeyboardInterrupt:
        control_sock.sendto(b"END", (SENDER_IP, SENDER_CONTROL_PORT))
        print("수신 연결 해제")
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()
        control_sock.close()
        wf.close()

if __name__ == "__main__":
    receive_audio()
