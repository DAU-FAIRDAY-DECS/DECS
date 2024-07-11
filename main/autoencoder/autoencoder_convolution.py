import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Dropout
from keras.models import Model

# 컨볼루션 오토인코더 정의
def convolutional_autoencoder():
    input_audio = Input(shape=(2048, 1))
    x = Conv1D(64, 3, activation='relu', padding='same')(input_audio)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    encoded = Conv1D(8, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
    return Model(input_audio, decoded)

# 데이터 로드 함수
def load_data(data_dir, target_length=2048):
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
                audio_data.append(y)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    print(f"Successfully loaded {len(audio_data)} files from {data_dir}.")
    return np.array(audio_data)

# 데이터 로드
train_normal_data = load_data('train_audio', 2048)
test_normal_data = load_data('test_audio', 2048)
test_abnormal_data = load_data('test_unaudio', 2048)

# 훈련 데이터 분할 (정상 데이터만 사용)
X_train, X_val = train_test_split(train_normal_data, test_size=0.2, random_state=42)

# 모델 훈련
model = convolutional_autoencoder()
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=5, batch_size=10, validation_data=(X_val, X_val))

# 테스트 데이터 준비 및 평가
X_test = np.concatenate((test_normal_data, test_abnormal_data))
y_test = np.array([0] * len(test_normal_data) + [1] * len(test_abnormal_data))
reconstructions = model.predict(X_test)
mse_test = np.mean(np.square(reconstructions - X_test), axis=(1, 2))

# 재구성 오류 히스토그램 비교
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
axes.hist(mse_test[y_test == 0], bins=50, alpha=0.75, label='Normal - Conv AE', range=(0, 3))
axes.hist(mse_test[y_test == 1], bins=50, alpha=0.75, label='Abnormal - Conv AE', range=(0, 3))
axes.set_title('Convolutional Autoencoder')
axes.set_xlabel('Reconstruction Error')
axes.set_ylabel('Number of Samples')
axes.legend()

plt.suptitle('Reconstruction Error Distribution - Conv AE')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 혼동 행렬 계산 및 시각화
threshold = np.percentile(mse_test, 95)
y_pred = (mse_test > threshold).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Normal', 'Abnormal'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# 재구성 오류 및 임계값 범위 탐색 그래프
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
axes.scatter(range(len(mse_test[y_test == 0])), mse_test[y_test == 0], c='blue', label='Normal - Conv AE')
axes.scatter(range(len(mse_test[y_test == 1])), mse_test[y_test == 1], c='orange', label='Abnormal - Conv AE')
#axes.fill_between(range(len(mse_test)), np.percentile(mse_test, 95), max(mse_test), color='orange', alpha=0.3, label='Threshold Range - Conv AE')
axes.set_title('Threshold Range Exploration - Conv AE')
axes.set_xlabel('Samples')
axes.set_ylabel('Reconstruction Error')
axes.set_ylim(0, 0.003)
axes.legend()

plt.tight_layout()
plt.show()

# 혼동 행렬 계산 및 시각화 (임계값 0.001)
threshold_fixed = 0.001
y_pred_fixed = (mse_test > threshold_fixed).astype(int)
conf_matrix_fixed = confusion_matrix(y_test, y_pred_fixed)
disp_fixed = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_fixed, display_labels=['Normal', 'Abnormal'])
disp_fixed.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix with Fixed Threshold (0.001)')
plt.show()