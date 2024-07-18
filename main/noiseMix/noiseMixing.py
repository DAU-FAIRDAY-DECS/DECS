from pydub import AudioSegment
import random

def mix_noise(original_audio_path, noise_audio_path, output_path):
    # 오디오 파일 로드
    original = AudioSegment.from_file(original_audio_path)
    noise = AudioSegment.from_file(noise_audio_path)

    # 원본 오디오 길이 계산
    original_length = len(original)

    # 노이즈를 믹싱할 횟수를 1부터 3까지 랜덤 설정
    num_noises = random.randint(1, 3)

    for _ in range(num_noises):
        # 노이즈의 지속 시간 설정 (0.1초에서 원본 길이의 5% 사이)
        noise_duration = random.randint(100, int(original_length * 0.05))
        
        # 노이즈 시작 지점 설정
        noise_start = random.randint(0, len(noise) - noise_duration)
        
        # 노이즈를 잘라내기
        noise_segment = noise[noise_start:noise_start + noise_duration]
        
        # 노이즈를 삽입할 원본 오디오의 위치 결정
        insert_position = random.randint(0, original_length - noise_duration)
        
        # 노이즈 믹싱
        original = original.overlay(noise_segment, position=insert_position)

    # 결과 저장
    original.export(output_path, format="wav")

# 예제 사용법
mix_noise("noiseMix/originals/음성원본.wav", "noiseMix/noises/배엔진소리.wav", "noiseMix/noiseMixedWav/output.wav")
