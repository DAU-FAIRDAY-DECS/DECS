import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Dropout
from keras.models import Model

# 컨볼루션 오토인코더 정의 (복잡성 증가 버전)
def convolutional_autoencoder():
    input_audio = Input(shape=(2048, 3))
    
    # 인코더
    x = Conv1D(128, 5, activation='relu', padding='same')(input_audio)  # 필터 수 128, 커널 크기 5
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # Dropout 증가
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)

    x = Conv1D(64, 5, activation='relu', padding='same')(x)  # 필터 수 64
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)

    x = Conv1D(32, 5, activation='relu', padding='same')(x)  # 필터 수 32
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(32, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)

    x = Conv1D(16, 5, activation='relu', padding='same')(x)  # 필터 수 16
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(16, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)

    encoded = Conv1D(8, 5, activation='relu', padding='same')(x)  # 더 깊이 추가된 인코더

    # 디코더
    x = Conv1D(16, 5, activation='relu', padding='same')(encoded)  # 필터 수 16
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(16, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)

    x = Conv1D(32, 5, activation='relu', padding='same')(x)  # 필터 수 32
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(32, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)

    x = Conv1D(64, 5, activation='relu', padding='same')(x)  # 필터 수 64
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)

    x = Conv1D(128, 5, activation='relu', padding='same')(x)  # 필터 수 128
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)

    decoded = Conv1D(3, 5, activation='sigmoid', padding='same')(x)

    return Model(input_audio, decoded)

# 특징 추출 함수
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

# 데이터 로드 함수
def load_data(data_dir, target_length=2048):
    audio_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(data_dir, filename)
            try:
                features = extract_features(file_path, target_length)
                audio_data.append(features)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    print(f"Successfully loaded {len(audio_data)} files from {data_dir}.")
    return np.array(audio_data)

# 데이터 로드
train_normal_data = load_data('train_audio', 2048)
test_normal_data = load_data('test_audio', 2048)
test_abnormal_data = load_data('test_unaudio', 2048)

# 데이터 차원 확장
train_normal_data = np.expand_dims(train_normal_data, axis=-1)
test_normal_data = np.expand_dims(test_normal_data, axis=-1)
test_abnormal_data = np.expand_dims(test_abnormal_data, axis=-1)

# 훈련 데이터 분할 (정상 데이터만 사용)
X_train, X_val = train_test_split(train_normal_data, test_size=0.2, random_state=42)

# 모델 훈련
model = convolutional_autoencoder()
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=10, batch_size=10, validation_data=(X_val, X_val))

# 테스트 데이터 준비 및 평가
X_test = np.concatenate((test_normal_data, test_abnormal_data))
y_test = np.array([0] * len(test_normal_data) + [1] * len(test_abnormal_data))
reconstructions = model.predict(X_test)
reconstructions = np.expand_dims(reconstructions, axis=-1)

mse_test = np.mean(np.square(reconstructions - X_test), axis=(1, 2))

# 재구성 오류 히스토그램 비교
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
axes.hist(mse_test[y_test == 0], bins=50, alpha=0.75, label='Normal - Conv AE', range=(0, 100))
axes.hist(mse_test[y_test == 1], bins=50, alpha=0.75, label='Abnormal - Conv AE', range=(0, 100))
axes.set_title('Convolutional Autoencoder')
axes.set_xlabel('Reconstruction Error')
axes.set_ylabel('Number of Samples')
axes.legend()

plt.suptitle('Reconstruction Error Distribution - Conv AE')
plt.show()

# 혼동 행렬 계산 및 시각화
threshold = np.percentile(mse_test, 95)
threshold_fixed = 60
y_pred = (mse_test > threshold_fixed).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Normal', 'Abnormal'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# # 임계값 임계값 임계값 임계값 임계값
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
# axes.scatter(range(len(mse_test[y_test == 0])), mse_test[y_test == 0], c='blue', label='Normal - Conv AE')
# axes.scatter(range(len(mse_test[y_test == 1])), mse_test[y_test == 1], c='orange', label='Abnormal - Conv AE')
# axes.fill_between(range(len(mse_test)), threshold_fixed, max(mse_test), color='orange', alpha=0.3, label='Threshold Range - Conv AE')
# axes.set_title('Threshold Range Exploration - Conv AE')
# axes.set_xlabel('Samples')
# axes.set_ylabel('Reconstruction Error')
# axes.legend()

# plt.tight_layout()
# plt.show()

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
    original_max = np.mean(np.abs(y_original))
    
    mse_error_rate = (mse / original_max) * 100  # MSE 오차율 (백분율)
    mae_error_rate = (mae / original_max) * 100  # MAE 오차율 (백분율)

    return mse_error_rate, mae_error_rate

# 원본 파일과 노이즈 섞인 파일 경로
original_wav_path = 't_error/original/input15.wav'
noisy_wav_path = 't_error/noise/output15.wav'

# 오차율 계산 (최대값 기준)
mse_error_rate, mae_error_rate = calculate_error_rate_max_based(original_wav_path, noisy_wav_path)

print(f"MSE Error Rate (Max-based): {mse_error_rate:.2f}%")
print(f"MAE Error Rate (Max-based): {mae_error_rate:.2f}%")