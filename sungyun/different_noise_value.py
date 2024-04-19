import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

def plot_time_series(data):
    fig = plt.figure(figsize=(10, 4))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()

data, sr = librosa.load('/Users/tjddbsj/Desktop/졸업프로젝트/무제/DECS/sungyun/new_voice_original.wav', sr=22050)

print(data.shape)
plot_time_series(data)  # 시간에 따른 소리 데이터의 변화를 그래프로 표시

## 부분적으로 가우시안 화이트 노이즈 삽입

def add_variable_white_noise(data, intervals, sr=22050):
    data_with_noise = np.copy(data)
    for start_time, end_time, noise_rate in intervals:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        data_with_noise[start_sample:end_sample] += noise_rate * np.random.randn(end_sample - start_sample)
    return data_with_noise

# 노이즈를 추가할 구간 설정: [(시작 시간, 끝 시간, 노이즈 레벨), ...]
intervals = [(5, 7, 0.05), (10, 11, 0.2), (15, 16, 0.1), (25, 27, 0.15), (31, 32, 0.3), (36, 38, 0.04)]

# 지정된 구간에 노이즈 추가
noisy_data = add_variable_white_noise(data, intervals)
plot_time_series(noisy_data)  # 노이즈가 추가된 데이터 플로팅

# 수정된 오디오 파일 저장
sf.write('add_noise_audio.wav', noisy_data, sr)
print('지정된 구간에 성공적으로 백색 소음 삽입')
