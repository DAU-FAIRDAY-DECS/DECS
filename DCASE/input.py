import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def load_mel_spectrogram(wav_file, n_mels=128, fmax=8000):
    # 오디오 파일 로드
    y, sr = librosa.load(wav_file, sr=None)
    # 멜 스펙트로그램 생성
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect_db

def save_melspectrogram(mel_spectrogram, sr, output_image):
    # mel_spectrogram이 1D일 경우, 이를 (n_mels, n_frames) 형태로 변환
    # n_mels와 n_frames를 적절히 설정합니다 (여기서는 예시로 n_mels=128을 사용)
    n_mels = 128  # 예시값, 필요에 따라 설정
    n_frames = len(mel_spectrogram) // n_mels  # mel_spectrogram 길이에 맞춰 프레임 수 계산
    
    if len(mel_spectrogram) % n_mels != 0:
        raise ValueError("mel_spectrogram의 길이가 n_mels로 나누어떨어지지 않습니다.")
    
    mel_spectrogram = mel_spectrogram.reshape(n_mels, n_frames)  # 1D를 2D로 변환
    
    # 멜 스펙트로그램을 시각화하여 저장
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()

# 모델 로드
model = tf.keras.models.load_model('model/model_human.hdf5')

# 입력 WAV 파일
input_wav = 'output9.wav'
output_image = 'reconstructed_melspectrogram.png'

# 입력 파일의 멜 스펙트로그램 생성
input_mel = load_mel_spectrogram(input_wav)

# 모델 입력 크기에 맞추기 위해 평탄화하고 필요한 경우 잘라내기
input_mel_flattened = input_mel.flatten()[:640]  # 모델이 (None, 640) 형식을 기대하는 경우
input_mel_expanded = np.expand_dims(input_mel_flattened, axis=0)  # 배치 차원 추가

# 모델을 통해 재구성된 멜 스펙트로그램 예측
reconstructed_mel = model.predict(input_mel_expanded)[0]

# 재구성된 멜 스펙트로그램을 이미지로 저장
save_melspectrogram(reconstructed_mel, sr=22050, output_image=output_image)
