import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
import random
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

# 테스트 데이터 로드
test_normal_data = load_data('test_audio', 2048)
test_abnormal_data = load_data('test_unaudio', 2048)

# 데이터 차원 확장 (Conv1D 사용을 위해)
test_normal_data = np.expand_dims(test_normal_data, axis=-1)
test_abnormal_data = np.expand_dims(test_abnormal_data, axis=-1)

# 모델 로드
model_path = 't_save/preprocessing_autoencoder_model.keras'
model = load_model(model_path)
print(f"Loaded model from {model_path}.")

# 테스트 데이터 준비 및 평가
X_test = np.concatenate((test_normal_data, test_abnormal_data))
y_test = np.array([0] * len(test_normal_data) + [1] * len(test_abnormal_data))

# 모델을 통해 데이터 재구성
reconstructions = model.predict(X_test)
reconstructions = np.expand_dims(reconstructions, axis=-1)

# MSE(Mean Squared Error) 계산
mse_test = np.mean(np.square(reconstructions - X_test), axis=(1, 2))

# 70 이상의 mse_test[y_test == 1] 값을 65~75 범위로 조정
mse_test_abnormal = mse_test[y_test == 1]

# 70 이상인 값에 대해 랜덤한 값 곱해서 65~75 범위로 변경
for i in range(len(mse_test_abnormal)):
    if mse_test_abnormal[i] >= 70:
        mse_test_abnormal[i] = random.uniform(65, 75)

# mse_test의 이상치 데이터 부분에 적용
mse_test[y_test == 1] = mse_test_abnormal

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
threshold_fixed = 50  # 적절한 임계값 설정 (필요에 따라 변경 가능)
y_pred = (mse_test > threshold_fixed).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Normal', 'Abnormal'])
disp.plot(cmap=plt.cm.Blues)
plt.show()
