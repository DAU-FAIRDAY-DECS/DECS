import wave
import random

def introduce_packet_issues(input_file, output_file, issue_rate=0.05, issue_severity='medium', max_issue_duration=1.0):
    """
    오디오 파일에서 화이트 노이즈를 랜덤하게 시뮬레이션

    :issue_rate: 각 프레임이 손상될 확률
    :issue_severity: 손상의 강도를 정의합니다 (low, medium, high)
    :max_issue_duration: 패킷 이슈가 최대 지속될 시간 (초 단위)
    """

    # 입력 WAV 파일 개방
    with wave.open(input_file, 'rb') as wav:
        # 파라미터 읽기
        params = wav.getparams()
        # 프레임 읽기
        frames = bytearray(wav.readframes(params.nframes))
        # 샘플 속도
        sample_rate = params.framerate
    
    # 강도에 따른 손상 값의 범위 설정
    if issue_severity == 'low':
        damage_range = (1, 50)
    elif issue_severity == 'medium':
        damage_range = (51, 150)
    elif issue_severity == 'high':
        damage_range = (151, 255)
    else:
        raise ValueError("Invalid issue_severity. Choose from 'low', 'medium', 'high'.")
    
    # 패킷 이슈 도입
    total_frames = len(frames)
    issue_duration_frames = int(sample_rate * max_issue_duration)
    
    # 랜덤한 시점에서 이슈를 삽입
    issue_start_frame = random.randint(0, total_frames - issue_duration_frames)
    issue_end_frame = issue_start_frame + issue_duration_frames
    
    for frame_index in range(issue_start_frame, issue_end_frame):
        # 이슈가 발생할 확률에 따라 프레임 손상
        if random.random() < issue_rate:
            frames[frame_index] = random.randint(damage_range[0], damage_range[1])
    
    # 이슈가 포함된 새 WAV 파일 작성
    with wave.open(output_file, 'wb') as new_wav:
        new_wav.setparams(params)
        new_wav.writeframes(frames)

# 파일 경로
input_wav_file = "main/voip/wav/0030.wav" # 입력 WAV 파일 경로
output_wav_file = "main/voip/sample/new/0030-white.wav" # 출력 WAV 파일 경로

# 패킷 이슈 도입
introduce_packet_issues(input_wav_file, output_wav_file, issue_rate=0.05, issue_severity='medium', max_issue_duration=1.0)