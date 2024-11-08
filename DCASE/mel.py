import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def save_melspectrogram(wav_file, output_image):
    # 오디오 파일 로드
    y, sr = librosa.load(wav_file, sr=None)

    # 멜 스펙트로그램 생성
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

    # 멜 스펙트로그램 시각화 및 저장
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()

# WAV 파일 경로와 출력 이미지 경로
wav_file_1 = 'input9.wav'
wav_file_2 = 'output9.wav'
output_image_1 = 'melspectrogram_1.png'
output_image_2 = 'melspectrogram_2.png'

# 각 파일의 멜 스펙트로그램 저장
save_melspectrogram(wav_file_1, output_image_1)
save_melspectrogram(wav_file_2, output_image_2)
