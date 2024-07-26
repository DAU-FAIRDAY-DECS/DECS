import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Dropout, LeakyReLU
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 오디오 데이터 정규화 함수
def normalize_audio(audio_data, global_mean, global_std):
    audio_data = (audio_data - global_mean) / global_std
    return audio_data

# 데이터를 로드하는 함수
def load_data(data_dir, target_length=2048, global_mean=None, global_std=None):
    audio_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(data_dir, filename)
            try:
                y, sr = librosa.load(file_path, sr=None)
                if len(y) < target_length:
                    y = np.pad(y, (0, target_length - len(y)), 'constant')
                else:
                    y = y[:target_length]
                y = np.expand_dims(y, axis=-1)
                if global_mean is not None and global_std is not None:
                    y = normalize_audio(y, global_mean, global_std)
                audio_data.append(y)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    print(f"Successfully loaded {len(audio_data)} files from {data_dir}.")
    return np.array(audio_data)

# 전체 데이터셋의 평균과 표준편차 계산
def calculate_global_mean_std(data_dirs, target_length=2048):
    all_data = []
    for data_dir in data_dirs:
        for filename in os.listdir(data_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(data_dir, filename)
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    if len(y) < target_length:
                        y = np.pad(y, (0, target_length - len(y)), 'constant')
                    else:
                        y = y[:target_length]
                    y = np.expand_dims(y, axis=-1)
                    all_data.append(y)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    all_data = np.concatenate(all_data, axis=0)
    global_mean = np.mean(all_data)
    global_std = np.std(all_data)
    return global_mean, global_std

# Convolutional Autoencoder 모델 정의
def convolutional_autoencoder():
    input_audio = Input(shape=(2048, 1))
    x = Conv1D(32, 3, padding='same')(input_audio)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Conv1D(32, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(64, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Conv1D(64, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    
    encoded = Conv1D(128, 3, padding='same')(x)
    encoded = LeakyReLU(alpha=0.1)(encoded)
    
    x = Conv1D(64, 3, padding='same')(encoded)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Conv1D(64, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    
    x = Conv1D(32, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    x = Conv1D(32, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
    return Model(input_audio, decoded)

# 데이터 디렉토리 목록
data_dirs = ['train_audio', 'test_audio', 'test_unaudio']

# 전체 데이터셋의 평균과 표준편차 계산
global_mean, global_std = calculate_global_mean_std(data_dirs)

# 데이터 로드
train_normal_data = load_data('train_audio', 2048, global_mean, global_std)
test_normal_data = load_data('test_audio', 2048, global_mean, global_std)
test_abnormal_data = load_data('test_unaudio', 2048, global_mean, global_std)

# 데이터 분할
X_train, X_val = train_test_split(train_normal_data, test_size=0.2, random_state=42)

# 모델 컴파일 및 훈련
model = convolutional_autoencoder()
model.compile(optimizer='adam', loss='mse')

# 콜백
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val), callbacks=[reduce_lr, early_stopping])

# 테스트 데이터 준비 및 평가
X_test = np.concatenate((test_normal_data, test_abnormal_data))
y_test = np.array([0] * len(test_normal_data) + [1] * len(test_abnormal_data))
reconstructions = model.predict(X_test)
mse_test = np.mean(np.square(reconstructions - X_test), axis=(1, 2))

# 재구성 오류 시각화
plt.figure(figsize=(10, 6))
plt.hist(mse_test[y_test == 0], bins=50, alpha=0.75, label='Normal')
plt.hist(mse_test[y_test == 1], bins=50, alpha=0.75, label='Abnormal')
plt.title('Reconstruction Error Histogram')
plt.xlabel('Reconstruction Error')
plt.ylabel('Number of Samples')
plt.legend()
plt.xlim(0, 50)  # 축 범위를 조정하여 정상 데이터가 잘 보이도록 설정
plt.show()

# 다양한 임계값을 테스트하여 최적의 값을 찾기 위한 코드
thresholds = np.arange(0.1, 1.0, 0.01)  # 0.1부터 1.0까지 0.01 간격으로 임계값 설정
best_threshold = 0
best_f1 = 0
f1_scores = []

for threshold in thresholds:
    y_pred = (mse_test > threshold).astype(int)
    current_f1 = f1_score(y_test, y_pred)
    f1_scores.append(current_f1)
    
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold

print(f"Best Threshold: {best_threshold}, Best F1 Score: {best_f1}")

# 재구성 오류 및 임계값 범위 탐색 그래프
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, marker='o')
plt.title('F1 Score vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.grid()
plt.show()

# 최적의 임계값을 사용하여 혼동행렬 계산 및 시각화
y_pred_best = (mse_test > best_threshold).astype(int)
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
disp_best = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_best, display_labels=['Normal', 'Abnormal'])
disp_best.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix with Best Threshold ({best_threshold})')
plt.show()
