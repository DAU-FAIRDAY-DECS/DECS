import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

# 첫 번째 음성 오디오 파일 로드
audio_file1 = 'yujin/wav/sy_original.wav'
y1, sr1 = librosa.load(audio_file1)

# 두 번째 음성 오디오 파일 로드
audio_file2 = 'yujin/wav/sy_noise.wav'
y2, sr2 = librosa.load(audio_file2)

# 첫 번째 음성의 멜 스펙트로그램 생성
S1 = librosa.feature.melspectrogram(y=y1, sr=sr1)

# 두 번째 음성의 멜 스펙트로그램 생성
S2 = librosa.feature.melspectrogram(y=y2, sr=sr2)

# 첫 번째 음성의 스펙트로그램을 dB로 변환
S1_db = librosa.power_to_db(S1, ref=np.max)

# 두 번째 음성의 스펙트로그램을 dB로 변환
S2_db = librosa.power_to_db(S2, ref=np.max)

# 첫 번째 음성의 멜 스펙트로그램 표시
plt.figure(figsize=(10, 4))
librosa.display.specshow(S1_db, sr=sr1, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (Audio 1)')
plt.show()

# 두 번째 음성의 멜 스펙트로그램 표시
plt.figure(figsize=(10, 4))
librosa.display.specshow(S2_db, sr=sr2, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram (Audio 2)')
plt.show()

# 두 개의 멜 스펙트로그램 이미지를 비교하는 함수
def compare_mel_spectrograms(S1, S2):
    # 유클리드 거리 계산
    euclidean_distance = euclidean(S1.flatten(), S2.flatten())
    
    # 피어슨 상관 계수 계산
    pearson_corr, _ = pearsonr(S1.flatten(), S2.flatten())
    
    # 구조적 유사성 지수 계산
    structural_similarity = ssim(S1, S2, data_range=S1.max() - S1.min())
    
    return euclidean_distance, pearson_corr, structural_similarity

# 두 개의 멜 스펙트로그램 이미지 비교
euclidean_dist, pearson_corr, ssim_index = compare_mel_spectrograms(S1_db, S2_db)

# 유사도 측정 함수
def calculate_similarity(euclidean_dist, pearson_corr, ssim_index):
    euclidean_min = 0  # 유클리드 거리 최소값
    euclidean_max = 20000  # 유클리드 거리 최대값
    pearson_min = -1  # 피어슨 상관 계수 최소값
    pearson_max = 1  # 피어슨 상관 계수 최대값
    ssim_min = 0  # 구조적 유사성 지수 최소값
    ssim_max = 1  # 구조적 유사성 지수 최대값

    # 각각의 지표 값 정규화
    normalized_euclidean = (euclidean_max - euclidean_dist) / (euclidean_max - euclidean_min)
    normalized_pearson = (pearson_corr - pearson_min) / (pearson_max - pearson_min)
    normalized_ssim = (ssim_index - ssim_min) / (ssim_max - ssim_min)

    # 임의의 가중치를 적용하여 유사도를 계산
    similarity = (normalized_euclidean * 0.4 + normalized_pearson * 0.3 + normalized_ssim * 0.3)

    # 유사도를 백분율로 변환
    similarity_percentage = similarity * 100

    return similarity_percentage

# 결과 출력
print("유클리드 거리:", euclidean_dist)
print("피어슨 상관 계수:", pearson_corr)
print("구조적 유사성 지수:", ssim_index)

# 유사도 측정
similarity_percentage = calculate_similarity(euclidean_dist, pearson_corr, ssim_index)

# 결과 출력
print("유사도: {:.2f}%".format(similarity_percentage))