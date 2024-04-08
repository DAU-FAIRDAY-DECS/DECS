import numpy as np
import librosa.display
import soundfile as sf

# 원본 WAV 파일 로드
original_wav, sr = librosa.load('yujin/wav/original.wav')

# 원본 WAV 파일의 길이 확인
original_length = len(original_wav)

# 노이즈 생성 (두 번째 인자가 곧 표준편차인데, 이걸 더 크게 조정하여 더 강한 노이즈를 생성할 수 있음)
strong_noise = np.random.normal(0, 0.01, original_length)

# 원본 WAV 파일에 노이즈 추가
noise_wav = original_wav + strong_noise

# 노이즈가 추가된 WAV 파일 저장
sf.write('yujin/wav/noise.wav', noise_wav, sr)