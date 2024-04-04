# 5초 동안 녹음 후 녹음 된 오디오를 스펙토그램으로 변환

import pyaudio
import numpy as np
import matplotlib.pyplot as plt

# 파이오디오 설정
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

# 파이오디오 객체 생성
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* 녹음을 시작합니다.")

frames = []

# 녹음 시작
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* 녹음이 완료되었습니다.")

# 스트림 정지 및 파이오디오 종료
stream.stop_stream()
stream.close()
p.terminate()

# 데이터를 numpy 배열로 변환
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

# FFT를 통한 주파수 변환
plt.specgram(audio_data, Fs=RATE, NFFT=1024, noverlap=512)

# 스펙트럼 시각화
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')
plt.show()
