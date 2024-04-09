import numpy as np
import random
import itertools
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import soundfile as sf  # soundfile 라이브러리를 추가

def plot_time_series(data):
    fig = plt.figure(figsize=(10, 4))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()

data, sr = librosa.load('taewon\wav\input.wav', sr=22050)

print(data.shape)
plot_time_series(data) 

# 원본 데이터

def adding_white_noise(data, sr=22050, noise_rate=0.005):
    # noise 방식으로 일반적으로 쓰는 잡음 끼게 하는 겁니다.
    wn = np.random.randn(len(data))
    data_wn = data + noise_rate*wn
    plot_time_series(data_wn)
    # librosa.output.write_wav 대신 soundfile 라이브러리의 write 함수를 사용합니다.
    sf.write('taewon\wav\white_noise.wav', data_wn, sr)  # 수정된 부분
    print('White Noise 저장 성공')
    
    return data

adding_white_noise(data)

# 원본 오디오 전체에 노이즈 삽입

def adding_white_noise_partially(data, start_time, end_time, sr=22050, noise_rate=0.2): # noise_rate 크기 조정 -> 노이즈 소리 조정
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # 오직 특정 구간에만 노이즈를 추가.
    data_with_noise = np.copy(data)
    data_with_noise[start_sample:end_sample] += noise_rate * np.random.randn(end_sample - start_sample)
    
    plot_time_series(data_with_noise)
    sf.write('taewon\wav\partial_white_noise.wav', data_with_noise, sr)
    print('부분적으로 White Noise 삽입 성공')
    
    return data_with_noise

# 예를 들어, 오디오의 5초부터 6초까지의 구간에 노이즈를 추가하고 싶다면:
adding_white_noise_partially(data, 5, 6)

# 일정 부분에 추가