import wave
import numpy as np
import matplotlib.pyplot as plt

def read_wave_file(filename):
    with wave.open(filename, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        return audio_data, framerate

def plot_waveforms(input_file, output_file):
    input_data, input_rate = read_wave_file(input_file)
    output_data, output_rate = read_wave_file(output_file)

    # x축을 시간 단위로 설정
    input_time = np.linspace(0, len(input_data) / input_rate, num=len(input_data))
    output_time = np.linspace(0, len(output_data) / output_rate, num=len(output_data))

    plt.figure(figsize=(15, 6))

    # input.wav 파형
    plt.subplot(2, 1, 1)
    plt.plot(input_time, input_data, label='Input Waveform', color='blue')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Input Waveform')
    plt.legend()

    # output.wav 파형
    plt.subplot(2, 1, 2)
    plt.plot(output_time, output_data, label='Output Waveform', color='orange')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Output Waveform')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 파일 경로 설정
input_file = 'main/voip/wav/input.wav'
output_file = 'main/voip/wav/output.wav'

# 파형 시각화
plot_waveforms(input_file, output_file)