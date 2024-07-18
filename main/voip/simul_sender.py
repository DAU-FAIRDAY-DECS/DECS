import pyaudio
import socket
import threading
import wave
import audioop
import random
import logging

# 로깅 설정
logging.basicConfig(filename='main/voip/log/sender.log', level=logging.DEBUG, filemode='w')

# 오디오 설정
FORMAT = pyaudio.paInt16 # 16비트 오디오 포맷
CHANNELS = 1 # 모노 채널
RATE = 8000 # 샘플링 레이트
CHUNK = 1024 # 버퍼당 프레임 수

# 포트 번호 및 수신자 IP 주소
PORT = 9001 # 통신 포트 번호
SENDER_CONTROL_PORT = 9002 # 송신자 제어 메시지 포트 번호
RECEIVER_CONTROL_PORT = 9003 # 수신자 제어 메시지 포트 번호
RECEIVER_IP = '192.168.25.3' # 수신자 IP 주소

def calculate_wav_length(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
        return duration

def send_audio():
    # WAV 파일의 길이 계산하여 패킷 손실 임계값 설정
    wav_length = calculate_wav_length("main/voip/wav/input.wav")
    # 초기값 0.3초와 전체 길이의 5%를 중 작은 값을 최대 지속 시간으로 설정
    max_loss_duration = min(0.3, 0.05 * wav_length)

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
    packet_id = 0
    last_loss_time = 0 # 마지막 패킷 손실 시간 기록

    try:
        while True:
            data = stream.read(CHUNK)
            wf.writeframes(data) # 입력 오디오 데이터를 파일에 저장

            # PCM 데이터를 ADPCM 데이터로 압축 (압축화)
            compressed_data, state = audioop.lin2adpcm(data, 2, state)
            
            # WAV 파일의 현재 시간 계산
            current_time = packet_id * (CHUNK / RATE)
            
            # 패킷 손실 시뮬레이션 (5% 확률로 약간의 빈 데이터 전송)
            if random.random() > 0.95 and (current_time - last_loss_time > max_loss_duration):
                # 전체 데이터의 20%만 빈 데이터로 채움
                empty_data_portion = int(CHUNK * 0.2)
                empty_data, _ = audioop.lin2adpcm(bytes([0] * empty_data_portion), 2, state)
                # UDP 소켓을 통해 패킷 단위로 전송 (패킷화)
                sock.sendto(empty_data, server_address)
                last_loss_time = current_time
                logging.debug(f'Sent partial empty packet (loss simulation) {packet_id} at WAV time {current_time:.3f} seconds')
            else:
                # UDP 소켓을 통해 패킷 단위로 전송 (패킷화)
                sock.sendto(compressed_data, server_address)
                logging.debug(f'Sent packet {packet_id} at WAV time {current_time:.3f} seconds')
            
            # 첫 번째 패킷 전송 시 연결 시작 메시지 전송
            if first_packet:
                control_sock.sendto(b"START", (RECEIVER_IP, RECEIVER_CONTROL_PORT))
                first_packet = False
            
            packet_id += 1
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
