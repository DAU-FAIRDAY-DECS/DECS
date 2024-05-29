import pyaudio
import socket
import threading
import wave
import audioop
import random
import time

# 오디오 설정
FORMAT = pyaudio.paInt16  # 16비트 오디오 포맷
CHANNELS = 1  # 모노 채널
RATE = 8000  # 샘플링 레이트
CHUNK = 1024  # 버퍼당 프레임 수

# 포트 번호 및 수신자 IP 주소
PORT = 9001 # 통신 포트 번호
SENDER_CONTROL_PORT = 9002  # 송신자 제어 메시지 포트 번호
RECEIVER_CONTROL_PORT = 9003  # 수신자 제어 메시지 포트 번호
RECEIVER_IP = '192.168.25.3'  # 수신자 IP 주소

def send_audio():
    # PyAudio 초기화 및 입력 스트림 열기
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    # UDP 소켓 생성
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_sock.bind(('0.0.0.0', SENDER_CONTROL_PORT))
    server_address = (RECEIVER_IP, PORT)
    
    print("오디오 송신 중")

    # 입력 오디오 데이터를 저장할 wave 파일 생성
    wf = wave.open("main/voip/wav/input.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    state = None # ADPCM 상태 변수

    def check_for_control_messages():
        while True:
            msg, _ = control_sock.recvfrom(1024)
            if msg.decode() == "END":
                break
        control_sock.close()
        print("오디오 송신 중지")
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
            wf.writeframes(data) # 입력 오디오 데이터를 파일에 저장

            # ADPCM 압축
            compressed_data, state = audioop.lin2adpcm(data, 2, state)
            
            # 패킷 손실 시뮬레이션 (10% 확률로 패킷 손실)
            if random.random() > 0.1:
                # 패킷 지연 시뮬레이션 (0-100ms 랜덤 지연)
                time.sleep(random.uniform(0, 0.1))
                sock.sendto(compressed_data, server_address)
            else:
                # 패킷 손실 시 빈 데이터 전송
                empty_data, _ = audioop.lin2adpcm(bytes([0] * CHUNK), 2, state)
                sock.sendto(empty_data, server_address)
                
            # 첫 번째 패킷 전송 시 연결 시작 메시지 전송
            if first_packet:
                control_sock.sendto(b"START", (RECEIVER_IP, RECEIVER_CONTROL_PORT))
                first_packet = False
    except KeyboardInterrupt:
        control_sock.sendto(b"END", (RECEIVER_IP, RECEIVER_CONTROL_PORT))
        print("오디오 송신 중지")
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()
        control_sock.close()
        wf.close()

if __name__ == "__main__":
    send_audio()