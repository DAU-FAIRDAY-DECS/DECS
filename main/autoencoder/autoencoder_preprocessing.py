import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import Model, load_model
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
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

# 데이터 차원 확장
train_normal_data = np.expand_dims(train_normal_data, axis=-1)

# 훈련 데이터 분할 (정상 데이터만 사용)
X_train, X_val = train_test_split(train_normal_data, test_size=0.2, random_state=42)

# 모델 저장 경로
model_path = 't_save/preprocessing_autoencoder_model.keras'

# 모델 로드 또는 생성
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Loaded model from disk.")
else:
    model = convolutional_autoencoder()
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 콜백
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 모델 훈련
    model.fit(X_train, X_train, epochs=50, batch_size=5, validation_data=(X_val, X_val), callbacks=[reduce_lr, early_stopping])
    
    # 모델 저장
    model.save(model_path)
    print(f"Model saved to {model_path}.")
