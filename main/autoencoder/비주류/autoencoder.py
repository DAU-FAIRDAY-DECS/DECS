import numpy as np
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Dropout, Concatenate, Lambda
from keras.models import Model

# 일반 오토인코더 정의
def autoencoder(input_dims):
    input_layer = Input(shape=(input_dims,))
    h = Dense(64, activation='relu')(input_layer)
    h = Dense(64, activation='relu')(h)
    h = Dense(8, activation='relu')(h)
    h = Dense(64, activation='relu')(h)
    h = Dense(64, activation='relu')(h)
    h = Dense(input_dims, activation=None)(h)
    return Model(inputs=input_layer, outputs=h, name='autoencoder')

# 잡음제거 오토인코더 정의
def noise_cancellation_autoencoder(input_dims):
    input_layer = Input(shape=(input_dims,))
    h = Dense(64, activation='relu')(input_layer)
    h = Dense(32, activation='relu')(h)
    h = Dense(16, activation='relu')(h)
    encoded = Dense(8, activation='relu')(h)
    h = Dense(16, activation='relu')(encoded)
    h = Dense(32, activation='relu')(h)
    h = Dense(64, activation='relu')(h)
    decoded = Dense(input_dims, activation=None)(h)
    return Model(inputs=input_layer, outputs=decoded, name='noise_cancellation_autoencoder')

# 이상신호 감지 오토인코더 정의
def abnormal_signal_detection_autoencoder(input_dims):
    input_layer = Input(shape=(input_dims,))
    h = Dense(64, activation='relu')(input_layer)
    h = Dense(32, activation='relu')(h)
    h = Dense(16, activation='relu')(h)
    encoded = Dense(8, activation='relu')(h)
    h = Dense(16, activation='relu')(encoded)
    h = Dense(32, activation='relu')(h)
    h = Dense(64, activation='relu')(h)
    decoded = Dense(input_dims, activation=None)(h)
    return Model(inputs=input_layer, outputs=decoded, name='abnormal_signal_detection_autoencoder')

# 다중 오토인코더 정의
def multi_autoencoder(input_dims, noise_cancel_dims, abnormal_dims):
    input_layer = Input(shape=(input_dims,))
    noise_cancel_input = Lambda(lambda x: x[:, :noise_cancel_dims])(input_layer)
    abnormal_input = Lambda(lambda x: x[:, noise_cancel_dims:])(input_layer)
    noise_cancel_autoencoder = noise_cancellation_autoencoder(noise_cancel_dims)
    noise_cancel_output = noise_cancel_autoencoder(noise_cancel_input)
    abnormal_signal_autoencoder = abnormal_signal_detection_autoencoder(abnormal_dims)
    abnormal_output = abnormal_signal_autoencoder(abnormal_input)
    merged_output = Concatenate()([noise_cancel_output, abnormal_output])
    decoded = Dense(input_dims, activation=None)(merged_output)
    return Model(inputs=input_layer, outputs=decoded, name='multi_autoencoder')

# 컨볼루션 오토인코더 정의
def convolutional_autoencoder():
    input_audio = Input(shape=(None, 1))
    x = Conv1D(64, 3, activation='relu', padding='same')(input_audio)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)  # Adding dropout
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)  # Adding dropout
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)  # Adding dropout
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)
    encoded = Conv1D(8, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)  # Adding dropout
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)  # Adding dropout
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)  # Adding dropout
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
    return Model(input_audio, decoded)

# 모델 초기화 및 출력
models = [
    autoencoder(100),
    noise_cancellation_autoencoder(100),
    abnormal_signal_detection_autoencoder(100),
    multi_autoencoder(100, 50, 50),
    convolutional_autoencoder()
]

# 모델 요약 출력
for model in models:
    model.summary()



########## 훈련 데이터 예시를 사용하여 일반 오토인코더, 다중 인코더 비교 ##########

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 데이터 생성 및 준비
data = np.random.normal(0, 1, (260, 100))  # 260개 샘플, 100개 특성
labels = np.array([0]*200 + [1]*60)  # 정상: 200개, 비정상: 60개

# 훈련 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 모델 훈련
# 일반 오토인코더
model_autoencoder = autoencoder(100)
model_autoencoder.compile(optimizer='adam', loss='mse')
model_autoencoder.fit(X_train, X_train, epochs=50, batch_size=10, validation_data=(X_test, X_test))

# 다중 오토인코더
model_multi_autoencoder = multi_autoencoder(100, 50, 50)
model_multi_autoencoder.compile(optimizer='adam', loss='mse')
model_multi_autoencoder.fit(X_train, X_train, epochs=50, batch_size=10, validation_data=(X_test, X_test))

# 재구성 오류 계산
reconstructions_auto = model_autoencoder.predict(X_test)
mse_auto = np.mean(np.square(reconstructions_auto - X_test), axis=1)

reconstructions_multi = model_multi_autoencoder.predict(X_test)
mse_multi = np.mean(np.square(reconstructions_multi - X_test), axis=1)

# 재구성 오류 히스토그램 비교
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), sharey=True)
axes[0].hist(mse_auto[y_test == 0], bins=50, alpha=0.75, label='Normal - Single AE')
axes[0].hist(mse_auto[y_test == 1], bins=50, alpha=0.75, label='Abnormal - Single AE')
axes[0].set_title('Single Autoencoder')
axes[0].set_xlabel('Reconstruction Error')
axes[0].set_ylabel('Number of Samples')
axes[0].legend()

axes[1].hist(mse_multi[y_test == 0], bins=50, alpha=0.5, label='Normal - Multi AE')
axes[1].hist(mse_multi[y_test == 1], bins=50, alpha=0.5, label='Abnormal - Multi AE')
axes[1].set_title('Multi Autoencoder')
axes[1].set_xlabel('Reconstruction Error')
axes[1].legend()

plt.suptitle('Reconstruction Error Distribution Comparison')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 임계값 설정 및 혼동 행렬 계산 비교
threshold_auto = np.percentile(mse_auto, 95)
y_pred_auto = [1 if e > threshold_auto else 0 for e in mse_auto]
conf_matrix_auto = confusion_matrix(y_test, y_pred_auto)

threshold_multi = np.percentile(mse_multi, 95)
y_pred_multi = [1 if e > threshold_multi else 0 for e in mse_multi]
conf_matrix_multi = confusion_matrix(y_test, y_pred_multi)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
disp_auto = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_auto, display_labels=['Normal', 'Abnormal'])
disp_auto.plot(ax=axes[0], cmap=plt.cm.Blues)
axes[0].set_title('Confusion Matrix - Single Autoencoder')

disp_multi = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_multi, display_labels=['Normal', 'Abnormal'])
disp_multi.plot(ax=axes[1], cmap=plt.cm.Blues)
axes[1].set_title('Confusion Matrix - Multi Autoencoder')

plt.tight_layout()
plt.show()

# 재구성 오류 및 임계값 범위 탐색 그래프
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
axes[0].scatter(range(len(mse_auto)), mse_auto, c='blue', label='Single AE Errors')
axes[0].fill_between(range(len(mse_auto)), np.percentile(mse_auto, 95), max(mse_auto), color='orange', alpha=0.3, label='Threshold Range - Single AE')
axes[0].set_title('Threshold Range Exploration - Single AE')
axes[0].set_xlabel('Samples')
axes[0].set_ylabel('Reconstruction Error')
axes[0].legend()

axes[1].scatter(range(len(mse_multi)), mse_multi, c='green', label='Multi AE Errors')
axes[1].fill_between(range(len(mse_multi)), np.percentile(mse_multi, 95), max(mse_multi), color='orange', alpha=0.3, label='Threshold Range - Multi AE')
axes[1].set_title('Threshold Range Exploration - Multi AE')
axes[1].set_xlabel('Samples')
axes[1].legend()

plt.tight_layout()
plt.show()

# 거짓양성 및 거짓음성률 그래프
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
thresholds_auto = np.linspace(min(mse_auto), max(mse_auto), 100)
fp_rates_auto = [np.sum((mse_auto > t) & (y_test == 0)) / np.sum(y_test == 0) for t in thresholds_auto]
fn_rates_auto = [np.sum((mse_auto <= t) & (y_test == 1)) / np.sum(y_test == 1) for t in thresholds_auto]

thresholds_multi = np.linspace(min(mse_multi), max(mse_multi), 100)
fp_rates_multi = [np.sum((mse_multi > t) & (y_test == 0)) / np.sum(y_test == 0) for t in thresholds_multi]
fn_rates_multi = [np.sum((mse_multi <= t) & (y_test == 1)) / np.sum(y_test == 1) for t in thresholds_multi]

axes[0].plot(thresholds_auto, fp_rates_auto, label='False Positive Rate - Single AE')
axes[0].plot(thresholds_auto, fn_rates_auto, label='False Negative Rate - Single AE')
axes[0].set_title('FP and FN Rates - Single AE')
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Rate')
axes[0].legend()

axes[1].plot(thresholds_multi, fp_rates_multi, label='False Positive Rate - Multi AE')
axes[1].plot(thresholds_multi, fn_rates_multi, label='False Negative Rate - Multi AE')
axes[1].set_title('FP and FN Rates - Multi AE')
axes[1].set_xlabel('Threshold')
axes[1].legend()

plt.tight_layout()
plt.show()