import pyaudio
import socket
import zlib
import threading
import wave
import random
import time

FORMAT = pyaudio.paInt16 # 오디오 포맷
CHANNELS = 1 # 오디오 채널 수
RATE = 8000 # 샘플링 레이트
CHUNK = 1024 # 처리할 프레임 수

PORT = 9001 # 포트 번호
SENDER_CONTROL_PORT = 9002 # 송신자 제어 메시지 포트 번호
RECEIVER_CONTROL_PORT = 9003 # 수신자 제어 메시지 포트 번호
SENDER_IP = '10.10.1.152' # 송신자 IP 주소

def receive_audio():
    # PyAudio 객체 생성 및 스트림 초기화
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
    
    # UDP 소켓 생성
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', PORT))
    control_sock.bind(('0.0.0.0', RECEIVER_CONTROL_PORT))
    
    print("수신 대기")
    
    # 출력 오디오 파일을 작성하기 위한 wave 객체 생성
    wf = wave.open("main/voip/wav/output.wav", 'wb')
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
            wf.writeframes(data) # 출력 오디오 데이터를 파일에 기록
            stream.write(data) # 오디오 데이터를 출력 장치로 전송

            # 다음 패킷 수신
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
