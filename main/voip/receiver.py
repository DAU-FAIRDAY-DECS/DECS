import pyaudio
import socket
import threading
import wave
import audioop
import random
import time

# 오디오 설정
FORMAT = pyaudio.paInt16 # 16비트 오디오 포맷
CHANNELS = 1 # 모노 채널
RATE = 8000 # 샘플링 레이트
CHUNK = 1024 # 버퍼당 프레임 수

# 포트 번호 및 송신자 IP 주소
PORT = 9001 # 통신 포트 번호
SENDER_CONTROL_PORT = 9002 # 송신자 제어 메시지 포트 번호
RECEIVER_CONTROL_PORT = 9003 # 수신자 제어 메시지 포트 번호
SENDER_IP = '192.168.25.3' # 송신자 IP 주소

def receive_audio():
    # PyAudio 초기화 및 출력 스트림 열기
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
    
    # UDP 소켓 생성
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', PORT))
    control_sock.bind(('0.0.0.0', RECEIVER_CONTROL_PORT))
    
    print("오디오 수신 대기 중")
    
    # 출력 오디오 데이터를 저장할 wave 파일 생성
    wf = wave.open("main/voip/wav/output.wav", 'wb')
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
        print("오디오 수신 중지")
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()
        wf.close()
        exit(0)

    control_thread = threading.Thread(target=check_for_control_messages)
    control_thread.start()

    try:
        # UDP 소켓을 통해 수신된 데이터를 ADPCM 데이터로 재조립 (패킷 재조립)
        compressed_data, addr = sock.recvfrom(2048)
        control_sock.sendto(b"START", (SENDER_IP, SENDER_CONTROL_PORT)) # 연결 시작 메시지 전송
        print("오디오 수신 중")
        while True:
            # 패킷 지연 시뮬레이션 (0-100ms 랜덤 지연)
            time.sleep(random.uniform(0, 0.1))
            
            # ADPCM 데이터를 PCM 데이터로 압축 해제 (압축 해제)
            data, state = audioop.adpcm2lin(compressed_data, 2, state)
            wf.writeframes(data) # 출력 오디오 데이터를 파일에 저장
            stream.write(data) # 오디오 데이터를 스피커로 출력
            
            # 다음 패킷 수신
            compressed_data, _ = sock.recvfrom(2048)
    except KeyboardInterrupt:
        control_sock.sendto(b"END", (SENDER_IP, SENDER_CONTROL_PORT))
        print("오디오 수신 중지")
        stream.stop_stream()
        stream.close()
        p.terminate()
        sock.close()
        control_sock.close()
        wf.close()

if __name__ == "__main__":
    receive_audio()