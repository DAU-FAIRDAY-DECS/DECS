import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 음성 오디오 파일 로드
audio_file = 'yujin/audio2.wav'
y, sr = librosa.load(audio_file)

# 멜 스펙트로그램 생성
S = librosa.feature.melspectrogram(y=y, sr=sr)

# 스펙트로그램을 dB로 변환
S_db = librosa.power_to_db(S, ref=np.max)

# 스펙트로그램 표시
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()
