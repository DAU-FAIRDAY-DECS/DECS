# # 16비트 정수 출력
# import pyaudio
# import numpy as np

# FORMAT = pyaudio.paInt16  # 오디오 포맷 설정 (16-bit PCM)
# CHANNELS = 1               # 단일 채널 (모노)
# RATE = 44100               # 샘플링 레이트 (44.1 kHz)
# CHUNK = 1024               # 오디오 버퍼 크기

# audio = pyaudio.PyAudio()

# # 오디오 스트림 열기
# stream = audio.open(format=FORMAT, channels=CHANNELS,
#                     rate=RATE, input=True,
#                     frames_per_buffer=CHUNK)

# print("녹음 및 디지털화 시작...")

# frames = []

# try:
#     # 오디오 데이터 읽어오기
#     while True:
#         data = stream.read(CHUNK)
#         frames.append(data)
        
#         # 데이터를 16-bit 정수형 배열로 변환
#         audio_data = np.frombuffer(data, dtype=np.int16)
        
#         # 여기서부터 디지털 데이터 처리 코드 작성 가능
#         print(audio_data)
        
# except KeyboardInterrupt:
#     print("녹음 및 디지털화 종료.")

# # 오디오 스트림 닫기
# stream.stop_stream()
# stream.close()
# audio.terminate()



# # 시각화 표현
# import pyaudio
# import numpy as np
# import matplotlib.pyplot as plt

# FORMAT = pyaudio.paInt16  # 오디오 포맷 설정 (16-bit PCM)
# CHANNELS = 1               # 단일 채널 (모노)
# RATE = 44100               # 샘플링 레이트 (44.1 kHz)
# CHUNK = 1024               # 오디오 버퍼 크기

# audio = pyaudio.PyAudio()

# # 오디오 스트림 열기
# stream = audio.open(format=FORMAT, channels=CHANNELS,
#                     rate=RATE, input=True,
#                     frames_per_buffer=CHUNK)

# print("녹음 및 디지털화 시작...")

# # 오디오 데이터 읽어오기
# data = stream.read(CHUNK)
# frames = [data]

# # 데이터를 16-bit 정수형 배열로 변환
# audio_data = np.frombuffer(data, dtype=np.int16)

# # 그래프 초기화
# plt.ion()
# fig, ax = plt.subplots()
# line, = ax.plot(audio_data)
# ax.set_xlabel('샘플 번호')
# ax.set_ylabel('오디오 신호 값')
# ax.set_title('오디오 파형 시각화')

# try:
#     while True:
#         # 오디오 데이터 읽어오기
#         data = stream.read(CHUNK)
#         frames.append(data)
        
#         # 데이터를 16-bit 정수형 배열로 변환
#         audio_data = np.frombuffer(data, dtype=np.int16)
        
#         # 그래프 업데이트
#         line.set_ydata(audio_data)
#         plt.draw()
#         plt.pause(0.01)  # 그래프가 업데이트되는 간격
        
# except KeyboardInterrupt:
#     print("녹음 및 디지털화 종료.")

# # 오디오 스트림 닫기
# stream.stop_stream()
# stream.close()
# audio.terminate()



# # 시각화 최적화
# import pyaudio
# import numpy as np
# import matplotlib.pyplot as plt

# FORMAT = pyaudio.paInt16  # 오디오 포맷 설정 (16-bit PCM)
# CHANNELS = 1               # 단일 채널 (모노)
# RATE = 44100               # 샘플링 레이트 (44.1 kHz)
# CHUNK = 1024               # 오디오 버퍼 크기

# audio = pyaudio.PyAudio()

# # 오디오 스트림 열기
# stream = audio.open(format=FORMAT, channels=CHANNELS,
#                     rate=RATE, input=True,
#                     frames_per_buffer=CHUNK)

# print("녹음 및 디지털화 시작...")

# # 그래프 초기화
# plt.ion()
# fig, ax = plt.subplots()
# x = np.arange(0, 2 * CHUNK, 2)
# line, = ax.plot(x, np.random.rand(CHUNK))
# ax.set_ylim(0, 255)
# ax.set_xlim(0, CHUNK)

# ax.set_xlabel('샘플 번호')
# ax.set_ylabel('오디오 신호 값')
# ax.set_title('오디오 파형 시각화')

# while True:
#     # 오디오 데이터 읽어오기
#     data = stream.read(CHUNK)
    
#     # 데이터를 16-bit 정수형 배열로 변환
#     audio_data = np.frombuffer(data, dtype=np.int16)
    
#     # 그래프 업데이트
#     line.set_ydata(audio_data)
#     plt.draw()
#     plt.pause(0.01)  # 그래프가 업데이트되는 간격

# # 오디오 스트림 닫기
# stream.stop_stream()
# stream.close()
# audio.terminate()



# # 시각화 둔감화 (최종)
import pyaudio
import numpy as np
import matplotlib.pyplot as plt

FORMAT = pyaudio.paInt16  # 오디오 포맷 설정 (16-bit PCM)
CHANNELS = 1               # 단일 채널 (모노)
RATE = 44100               # 샘플링 레이트 (44.1 kHz)
CHUNK = 1024               # 오디오 버퍼 크기

audio = pyaudio.PyAudio()

# 오디오 스트림 열기
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("녹음 및 디지털화 시작...")

# 그래프 초기화
plt.ion()
fig, ax = plt.subplots()
x = np.arange(0, 2 * CHUNK, 2)
line, = ax.plot(x, np.random.rand(CHUNK))
ax.set_ylim(-2000, 2000)  # y축 범위 조정
ax.set_xlim(0, CHUNK)

ax.set_xlabel('샘플 번호')
ax.set_ylabel('오디오 신호 값')
ax.set_title('오디오 파형 시각화')

while True:
    # 오디오 데이터 읽어오기
    data = stream.read(CHUNK)
    
    # 데이터를 16-bit 정수형 배열로 변환
    audio_data = np.frombuffer(data, dtype=np.int16)
    
    # 그래프 업데이트
    line.set_ydata(audio_data)
    plt.draw()
    plt.pause(0.01)  # 그래프가 업데이트되는 간격

# 오디오 스트림 닫기
stream.stop_stream()
stream.close()
audio.terminate()
