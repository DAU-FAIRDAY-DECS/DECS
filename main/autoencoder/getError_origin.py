import numpy as np
import librosa
import os

# WAV 파일 간의 재구성 오차(오차율) 계산 함수 (최대값 기준)
def calculate_error_rate_max_based(original_wav_path, noisy_wav_path):
    # 두 파일 로드
    y_original, sr_original = librosa.load(original_wav_path, sr=None)
    y_noisy, sr_noisy = librosa.load(noisy_wav_path, sr=None)

    # 두 파일의 샘플링 레이트가 다를 경우 재샘플링
    if sr_original != sr_noisy:
        y_noisy = librosa.resample(y_noisy, sr_noisy, sr_original)

    # 두 파일의 길이가 다를 경우, 짧은 파일 기준으로 길이를 맞춤
    min_len = min(len(y_original), len(y_noisy))
    y_original = y_original[:min_len]
    y_noisy = y_noisy[:min_len]

    # MSE (Mean Squared Error) 계산
    mse = np.mean(np.square(y_original - y_noisy))

    # MAE (Mean Absolute Error) 계산
    mae = np.mean(np.abs(y_original - y_noisy))

    # 오차율 계산 (원본 데이터의 최대값을 기준으로)
    original_max = np.max(np.abs(y_original))

    mse_error_rate = (mse / original_max) * 100  # MSE 오차율 (백분율)
    mae_error_rate = (mae / original_max) * 100  # MAE 오차율 (백분율)

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

# 원본 파일과 노이즈 섞인 파일 경로
original_wav_path = 't_error/original/input3.wav'
noisy_wav_path = 't_error/noise/output3.wav'

# 오차율 계산 (최대값 기준)
mse_error_rate, mae_error_rate = calculate_error_rate_max_based(original_wav_path, noisy_wav_path)

# 결과 출력
print(f"MSE Error Rate (Max-based): {mse_error_rate:.2f}%")
print(f"MAE Error Rate (Max-based): {mae_error_rate:.2f}%")
