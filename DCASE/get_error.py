import numpy as np
import librosa
import os
from keras.models import load_model
import common as com  # 원래 common.py에서 제공한 기능들 사용

# 모델 경로
model_path = 'model/model_human.keras'
# 테스트할 파일 경로
wav_path = 'dev_data\\human\\test\\0001-loss.wav' #44.282575952544455

# 파일을 100개 구간으로 나눠서 재구성 오차 계산 함수
def calculate_segmented_reconstruction_error(wav_path, model, num_segments=100, threshold=0.5, n_mels=128, n_frames=640, n_fft=1024, hop_length=512, power=2.0):
    # 파일 로드
    y, sr = librosa.load(wav_path, sr=None)
    segment_length = len(y) // num_segments  # 각 구간 길이
    segment_errors = []

    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = y[start:end]

        # STFT와 멜 스펙트로그램 변환
        mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=power)
        
        # mel_spectrogram의 크기를 모델 입력 크기와 맞추기 (n_mels * n_frames)
        mel_spectrogram = mel_spectrogram.flatten()  # (n_mels * n_frames) 형태로 변환
        if len(mel_spectrogram) < n_frames:
            mel_spectrogram = np.pad(mel_spectrogram, (0, n_frames - len(mel_spectrogram)), mode='constant')
        else:
            mel_spectrogram = mel_spectrogram[:n_frames]  # 필요한 만큼 자르기

        # 모델 입력 형태에 맞게 차원 확장 (배치 크기 추가)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # (1, 640) 형태로 변환

        # 모델을 통한 재구성 및 오차 계산
        reconstruction = model.predict(mel_spectrogram)
        reconstruction = np.squeeze(reconstruction)  # 차원 축소
        reconstruction_error = np.mean(np.square(reconstruction - mel_spectrogram))

        # 임계값 비교 후 결과 저장
        segment_errors.append(reconstruction_error > threshold)
        print(f"Segment {i+1} reconstruction_error = {reconstruction_error}")

    return segment_errors

# 모델 로드
if os.path.exists(model_path):
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
else:
    print(f"Model file {model_path} does not exist!")
    exit()

# 재구성 오차 계산
segment_errors = calculate_segmented_reconstruction_error(wav_path, model)

# 결과 출력
for i, error in enumerate(segment_errors):
    status = "Anomalous" if error else "Normal"
    print(f"Segment {i + 1}: {status}")
