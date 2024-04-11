import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def wav_to_image(input_wav_file, output_image_file, dpi=300):   #dpi 조정가능
    # WAV 파일 로드
    sample_rate, data = wavfile.read(input_wav_file)

    # 데이터가 2차원이 아닌 경우, 즉 모노 또는 스테레오가 아닌 경우 처리
    if len(data.shape) > 1:
        data = data.sum(axis=1) / 2

    # 데이터의 크기
    num_samples = len(data)

    # 이미지 생성
    plt.figure(figsize=(num_samples / sample_rate, 2))
    plt.plot(np.linspace(0, num_samples / sample_rate, num_samples), data, color='black')
    plt.axis('off')
    plt.savefig(output_image_file, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()

# WAV 파일을 이미지로 변환
wav_to_image("taewon\wav\input.wav", "taewon\png\input.png")
wav_to_image("taewon\wav\partial_white_noise.wav", "taewon\png\partial_white_noise.png")
wav_to_image("taewon\wav\white_noise.wav", "taewon\png\white_noise.png")