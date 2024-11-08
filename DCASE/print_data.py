import numpy as np
import librosa
import keras
import keras_model
import common as com

# 1. WAV 파일 로드 (common.py에 있는 함수 사용)
def load_wav(file_path, sr=8000):
    audio, _ = librosa.load(file_path, sr=sr)  # sr은 샘플링 주파수 (여기선 8000으로 설정)
    return audio

# 2. 데이터를 멜 스펙트로그램으로 변환 (common.py에 있는 함수 사용)
def audio_to_mel_spectrogram(file_path, n_mels=128, n_frames=5, n_fft=1024, hop_length=512):
    mel_spectrogram = com.file_to_vectors(file_path,
                                          n_mels=n_mels,
                                          n_frames=n_frames, 
                                          n_fft=n_fft,
                                          hop_length=hop_length,
                                          power=2.0)
    return mel_spectrogram


def reconstruct_data(model, mel_spectrogram):
    mel_spectrogram_reshaped = np.expand_dims(mel_spectrogram, axis=-1)  # 모델 입력에 맞게 배치 차원 추가
    reconstructed = model.predict(mel_spectrogram_reshaped)
    return reconstructed[0]  # 배치 차원 제거하고 원본 형태로 반환

# 4. 원본 데이터와 재구성된 데이터 비교 (재구성 오차 계산)
def calculate_reconstruction_error(original, reconstructed):
    error = np.mean(np.square(original - reconstructed))  # MSE 계산
    return error

# 예시 실행
if __name__ == "__main__":
    wav_file_path = 'dev_data\\human\\test\\0001-loss.wav' #44.282575952544455
    model_path = "model\model_human_100_64.keras"  # 학습된 모델 경로

    # 1. WAV 파일 로드
    original_audio = load_wav(wav_file_path)

    # 2. 원본 오디오 데이터를 멜 스펙트로그램으로 변환
    original_data = audio_to_mel_spectrogram(wav_file_path)

    # 3. 오토인코더 모델 로드
    model = keras_model.load_model(model_path)

    # 4. 데이터를 재구성
    reconstructed_data = reconstruct_data(model, original_data)

    # 5. 재구성 오차 계산
    reconstruction_error = calculate_reconstruction_error(original_data, reconstructed_data)
    
    # 결과 출력
    print("Original Data (Mel Spectrogram):")
    print(original_data)
    print("\nReconstructed Data (Mel Spectrogram):")
    print(reconstructed_data)
    print("\nReconstruction Error:", reconstruction_error)
