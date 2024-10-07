import numpy as np
import librosa
import os
from keras.models import load_model

# 모델 경로
model_path = 't_save/preprocessing_autoencoder_model.h5'

# WAV 파일 간의 재구성 오차(오차율) 계산 함수 (최대값 기준)
def calculate_error_rate_max_based(noisy_wav_path, model):
    # 파일 로드
    y_noisy, sr_noisy = librosa.load(noisy_wav_path, sr=None)

    # 모델을 통해 노이즈 파일 재구성
    # 특징 추출 및 차원 확장
    features_noisy = extract_features(noisy_wav_path, 2048)
    features_noisy = np.expand_dims(features_noisy, axis=0)
    features_noisy = np.expand_dims(features_noisy, axis=-1)
    y_noisy = np.reshape(y_noisy, (-1, 1, 1, 1))

    # 모델을 통해 데이터 재구성
    reconstructions = model.predict(features_noisy)
    reconstructions = np.squeeze(reconstructions)
    
    # MSE (Mean Squared Error) 계산
    mse = np.mean(np.square(y_noisy - reconstructions))

    # MAE (Mean Absolute Error) 계산
    mae = np.mean(np.abs(y_noisy - reconstructions))

    # 오차율 계산 (원본 데이터의 최대값을 기준으로)
    noisy_max = np.max(np.abs(y_noisy))

    mse_error_rate = (mse / noisy_max) * 100  # MSE 오차율 (백분율)
    mae_error_rate = (mae / noisy_max) * 100  # MAE 오차율 (백분율)

    return mse_error_rate, mae_error_rate

# 특징 추출 함수 (기존과 동일)
def extract_features(file_path, target_length=2048):
    y, sr = librosa.load(file_path, sr=None)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), 'constant')
    else:
        y = y[:target_length]
    energy = np.square(y)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # 모든 특징의 길이를 맞추기 위해 자르거나 패딩
    energy = np.pad(energy, (0, target_length - len(energy)), 'constant')
    pitch = np.pad(pitch, (0, target_length - len(pitch)), 'constant')
    zcr = np.pad(zcr, (0, target_length - len(zcr)), 'constant')

    # 특징 결합
    features = np.vstack((energy, pitch, zcr)).T
    return features

# 노이즈 섞인 파일 경로
noisy_wav_path = 't_error/noise/output.wav'

# 모델 로드
if os.path.exists(model_path):
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
else:
    print(f"Model file {model_path} does not exist!")
    exit()

# 오차율 계산 (최대값 기준)
mse_error_rate, mae_error_rate = calculate_error_rate_max_based(noisy_wav_path, model)

# 결과 출력
print(f"MSE Error Rate (Max-based): {mse_error_rate:.2f}%")
print(f"MAE Error Rate (Max-based): {mae_error_rate:.2f}%")
