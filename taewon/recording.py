import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# 파이오디오 설정
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 9
WAVE_OUTPUT_FILENAME = "output.wav"

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

# Waveplot 시각화
plt.figure(figsize=(12, 4))
plt.plot(audio_data)
plt.title('Waveplot')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# WAV 파일로 저장
sf.write(WAVE_OUTPUT_FILENAME, audio_data, RATE)
print(f"* '{WAVE_OUTPUT_FILENAME}' 파일로 저장되었습니다.")