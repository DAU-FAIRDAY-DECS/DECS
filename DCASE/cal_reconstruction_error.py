import os
import numpy as np
import scipy.stats
import joblib
import keras_model
import common as com

# 필요한 파라미터 로드
param = com.yaml_load()

def calculate_reconstruction_error(wav_path, model, n_mels=64, n_frames=512, n_fft=1024, hop_length=512, power=2.0):
    try:
        # WAV 파일을 벡터로 변환 (멜 스펙트로그램으로 변환)
        data = com.file_to_vectors(wav_path,
                                    n_mels=n_mels,
                                    n_frames=n_frames,
                                    n_fft=n_fft,
                                    hop_length=hop_length,
                                    power=power)
        
        print(f"Data shape: {data.shape}")
        print(f"Data mean: {np.mean(data)}, std: {np.std(data)}")
        
        # 모델 예측 수행
        reconstruction = model.predict(data)
        
        print(f"Reconstruction shape: {reconstruction.shape}")
        print(f"Reconstruction mean: {np.mean(reconstruction)}, std: {np.std(reconstruction)}")
        
        # 재구성 오차 계산 (MSE 방식)
        reconstruction_error = np.mean(np.square(data - reconstruction))
        
        if np.isnan(reconstruction_error) or np.isinf(reconstruction_error):
            print("Invalid reconstruction error")
            return None
        
        return reconstruction_error
    except Exception as e:
        print(f"Error processing file {wav_path}: {e}")
        return None


if __name__ == "__main__":
    # WAV 파일 경로 설정
    wav_path = "output9.wav"  # 여기에 실제 파일 경로를 입력하세요

    # 모델 로드
    model_file = "{model}/model_human.keras".format(model=param["model_directory"])
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found!")
        exit(-1)
    
    model = keras_model.load_model(model_file)
    model.summary()

    # 재구성 오차 계산
    error = calculate_reconstruction_error(wav_path, model)
    
    if error is not None:
        print(f"Reconstruction Error for {wav_path}: {error}")
    else:
        print("Failed to calculate reconstruction error.")
    
    # 모델과 세션 정리
    keras_model.clear_session()
