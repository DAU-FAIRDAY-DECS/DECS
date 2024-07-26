import random
from pydub import AudioSegment
import os

def simulate_random_packet_loss(audio, min_loss_length_ms=100, max_loss_length_ms=500, loss_probability=0.1):
    """
    오디오 파일에서 패킷 손실을 랜덤하게 시뮬레이션

    :param audio: 원본 오디오를 포함하는 AudioSegment 객체
    :param min_loss_length_ms: 손실의 최소 길이 (밀리초 단위)
    :param max_loss_length_ms: 손실의 최대 길이 (밀리초 단위)
    :param loss_probability: 주어진 밀리초에서 손실이 발생할 확률
    :return: 시뮬레이션된 패킷 손실이 포함된 AudioSegment
    """

    length_ms = len(audio) # 오디오의 전체 길이를 밀리초 단위로 측정
    output_audio = AudioSegment.silent(duration=0) # 출력 오디오를 초기화 (무음)
    index = 0 # 현재 위치를 나타내는 인덱스

    while index < length_ms:
        if random.random() < loss_probability: # 설정된 확률에 따라 손실 발생 여부 결정
            # 손실 지속 시간을 랜덤하게 계산
            loss_duration = random.randint(min_loss_length_ms, max_loss_length_ms)
            # 지속 시간 동안 무음 추가
            output_audio += AudioSegment.silent(duration=loss_duration)
            # 원본 오디오에서 손실 부분 건너뛰기
            index += loss_duration
        else:
            # 원본 파일에서 랜덤한 길이의 오디오 추가
            next_audio_length = random.randint(1, 100) # 1ms에서 100ms 사이의 랜덤 길이
            output_audio += audio[index:index + next_audio_length]
            index += next_audio_length

    return output_audio

def process_all_audio_files(input_directory, output_directory):
    # 지정된 폴더의 모든 WAV 파일을 찾아서 처리
    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_directory, filename)
            output_filename = filename.split('.')[0] + '-loss.wav'
            output_path = os.path.join(output_directory, output_filename)

            # 오디오 파일 로드
            original_audio = AudioSegment.from_file(file_path, format="wav")

            # 패킷 손실 시뮬레이션 적용
            audio_with_packet_loss = simulate_random_packet_loss(original_audio)

            # 조작된 오디오 내보내기
            audio_with_packet_loss.export(output_path, format="wav")

# input_directory와 output_directory 경로 설정
input_directory = "main/voip/wav"
output_directory = "main/voip/sample/new"

# 모든 파일에 대한 처리 시작
process_all_audio_files(input_directory, output_directory)