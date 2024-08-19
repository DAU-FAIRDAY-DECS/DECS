import librosa
import numpy as np

def extract_audio_features(file_path):
    # WAV 파일 로드
    y, sr = librosa.load(file_path, sr=None)

    # 프레임 크기와 홉 크기 정의
    frame_length = 1024
    hop_length = 512

    # 시간 도메인 특징 추출
    # 1. 에너지
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    np.savetxt("main/autoencoder/feature/1에너지.txt", energy.T, fmt='%.4f')

    # 2. 영점 교차율 (Zero Crossing Rate)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)
    np.savetxt("main/autoencoder/feature/2교차율.txt", zcr.T, fmt='%.4f')

    # 3. 기본 주파수 (F0)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    np.savetxt("main/autoencoder/feature/3주파수.txt", f0, fmt='%.4f', delimiter='\n')

    # 주파수 도메인 특징 추출
    # 1. 스펙트럼 중심
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    np.savetxt("main/autoencoder/feature/4중심.txt", spectral_centroids.T, fmt='%.4f')

    # 2. 스펙트럼 대역폭
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    np.savetxt("main/autoencoder/feature/5대역.txt", spectral_bandwidth.T, fmt='%.4f')

    # 3. 스펙트럼 롤오프
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
    np.savetxt("main/autoencoder/feature/6롤옾.txt", spectral_rolloff.T, fmt='%.4f')

if __name__ == "__main__":
    extract_audio_features("00003.wav")
