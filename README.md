# DECS
### 디지털화된 오디오를 네트워크로 전송시 에러율 확인 소프트웨어
>**D**etection **E**rror in **C**ommunication **S**ystem
- 소속 : 동아대학교 컴퓨터공학과<br/> 
- 개발기간 : 2024.03 ~ 2024.12<br/> 

### 팀원

| 이름 | 역할 |
|-----------|-----------|
| 정성윤(팀장) | Autoencoder 이상 감지 모델 구성 및 학습 |
| 강태원 | Autoencoder 이상 감지 모델 구성 및 학습 |
| 박유진 | UDP 기반 VoIP 프로토콜 통신 및 시각화 |
| 전민재 | UDP 기반 VoIP 프로토콜 통신 및 시각화 |

---

## 프로젝트 소개

디지털 오디오 네트워크 전송 에러율 확인 소프트웨어는 네트워크를 통해 전송된 오디오 데이터의 에러율을 감지하고 분석하여, 네트워크 전송의 안정성을 향상시키는 것을 목표로 함.

## 개발 배경 및 필요성

- 네트워크를 통해 전송되는 오디오 데이터의 품질은 **정확한 정보 전달**을 위해 필수적임.  
- 전송 과정에서 발생하는 **노이즈와 패킷 이슈**는 데이터 왜곡 및 정보 손실을 초래할 수 있음.  
- 특히 **함정과 같은 특수 환경**에서는 통신 장애가 안전사고로 이어질 수 있어, **정확한 에러 감지와 분석**이 필요함.  

## 주요 기능

1. **VoIP 프로토콜 통신**
   - UDP 기반 VoIP 환경에서 패킷 손실, 지연, 순서 변경 등을 시뮬레이션함.
   - 이를 통해 정상 및 비정상 테스트 데이터를 확보함.

2. **Anomaly Detection**
   - Autoencoder 기반 모델을 사용해 정상 데이터와 비정상 데이터를 구분함.
   - 모델의 재구성 오차를 바탕으로 에러율을 감지하고, F1 Score로 성능을 평가함.

3. **에러율 검출**  
   - 재구성 오차를 활용하여 백분율로 에러율을 계산함.  
   - 이를 기반으로 **사전 예방 및 통신 안정성 향상**을 기대할 수 있음. 

---

## 시스템 구조

<img src="https://github.com/user-attachments/assets/e3b4c2e4-6aaf-492c-8ac0-8283c2458a0b" width="600">

1. **데이터 생성**: VoIP 환경에서 정상/비정상 오디오 데이터 생성함.
2. **데이터 학습**: Autoencoder 모델에 정상 데이터를 학습시킴.
3. **에러율 출력**: 학습된 모델로 비정상 데이터를 판별하고 에러율을 계산함.

## VoIP 프로토콜 통신

- **패킷 전송**: UDP 기반 네트워크 환경에서 데이터를 전송함.  
- **패킷 이슈 시뮬레이션**: 손실, 지연, 순서 변경 등을 추가하여 에러율을 극대화함. 

## 송수신 서버

<img src="https://github.com/user-attachments/assets/aec16782-2ada-4e31-bde6-220642cdc14c" width="600"/>

### 송신자

1. PyAudio 초기화 및 오디오 입력 스트림 개방
2. UDP 소켓 생성
3. PCM 데이터를 ADPCM 데이터로 압축
4. 패킷 이슈 시뮬레이션
5. UDP 소켓을 통한 데이터 전송

### 수신자

1. PyAudio 초기화 및 오디오 출력 스트림 개방
2. UDP 소켓 생성
3. 수신된 데이터를 ADPCM 데이터로 재조립
4. 패킷 이슈 시뮬레이션
5. ADPCM 데이터를 PCM 데이터로 압축 해제

## 스펙트로그램

<img src="https://github.com/user-attachments/assets/e336881c-f056-4116-94ce-cb30e2730c80" width="1000">

- WAV 파일을 로드하고 그 파일의 멜 스펙트로그램을 생성함.
- 음성 인터넷 프로토콜 환경에서 패킷 이슈를 인위적으로 발생시킴함.
- 이러한 과정을 반복해 테스트를 위한 정상/비정상 음성 샘플을 확보함.

## WAV 데이터 수집

<img src="https://github.com/user-attachments/assets/c9570755-fc26-4c28-945f-b9913bf72917" width="600">

- **러닝 데이터**: 노이즈가 없는 정상 데이터로 총 13,500개.
- **테스트 데이터**: 정상 481개, 비정상 461개로 총 942개.

## Autoencoder 구조

### **Encoder**
- 입력 데이터(128 차원)를 점진적으로 축소하여 최종적으로 **Latent Space**(8 차원)로 압축함.
- 주요 차원 변화:  
  - **128 → 64 → 32 → 16 → 8**

### **Latent Space**
- 데이터를 압축하여 중요한 특징만 포함된 공간(8 차원)임.

### **Decoder**
- Latent Space 데이터를 기반으로 원본 데이터(128 차원)로 점진적으로 복원함.
- 주요 차원 변화:  
  - **8 → 16 → 32 → 64 → 128**

## Autoencoder 입출력 데이터 텐서

<img src="https://github.com/user-attachments/assets/f17b2610-4618-4fc7-a25e-ff10a1a06aba" width="1000">

- **특징 벡터 생성**: 여러 프레임을 하나의 벡터로 이어붙여 **320차원의 특징 벡터**를 생성함함.

### 1. 정상 데이터 텐서

- **정상 데이터**는 오토인코더 모델 학습에 사용됨.
- **구성**: 총 97개의 데이터로 이루어진 320차원의 텐서임.
- 패킷 이슈 없이 안정적인 값을 보여줌.

### 2. 비정상 데이터 텐서
- **비정상 데이터**는 패킷 손실 등 이슈가 삽입된 데이터임.
- **이슈 집중**: 특히 96행과 97행에 노이즈가 포함되어 데이터 값이 비정상적으로 변화함.

### 3. 비정상 데이터 재구성 텐서
- 오토인코더를 사용해 비정상 데이터를 재구성한 결과, 정상 데이터와 유사한 형태로 복원됨.
- **특징**: 재구성 텐서는 비정상 데이터와 비교하여 오차가 줄어들었음을 확인할 수 있음.
  
## 모델 성능 평가
<img alt="image" src="https://github.com/user-attachments/assets/bbab3938-de19-4c54-9329-d48c5c418aa1" width="400">

> 혼동행렬

### 정확도 (Accuracy)
정확도는 모델이 정확하게 예측한 샘플의 비율을 나타내는 지표.

### 정밀도 (Precision)
정밀도는 모델이 양성으로 예측한 샘플 중 실제로 양성인 샘플의 비율을 나타내는 지표.

### 재현률 (Recall)
재현률은 실제 양성인 샘플 중에서 모델이 올바르게 양성으로 예측한 비율을 나타내는 지표.

### F1 Score
F1 Score은 모델의 예측 성능을 종합적으로 평가하는 지표.

|                 | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| Actual Positive | TP = 416           | FN = 198           |
| Actual Negative | FP = 45            | TN = 283           |
- **정확도**: 74%
- **정밀도**: 90%
- **재현율**: 68%
- **F1 Score**: 77%

### 임계값 수정에 따른 F1 Score 변화

<img src="https://github.com/user-attachments/assets/79f4ae64-9b9a-4bf9-812c-26e3ea96fefa" width="600">

- 임계값이 0.85일 때, 가장 높은 F1 Score가 나타남.

## 에러율 검출

1. **최대 오차 대비 백분율**
   - 재구성 오차를 최대 허용 오차로 나누어 백분율 계산.
2. **평균 오차 대비 백분율**
   - 재구성 오차를 정상 데이터의 평균 오차로 나누어 백분율 계산.
3. **정규화 오차 대비 백분율**
   - 재구성 오차를 최소~최대 오차 범위 내에서 정규화하여 백분율 계산.
  
- 최대와 최소를 활용한 방법은 값의 편차가 클 경우 예상치 못한 값이 나올 수 있다는 한계가 있기에, **안정적인 에러율 출력을 보장**할 수 있는 **평균 오차 대비 백분율** 방식을 채택함.

### 평균 오차 대비 백분율

<img src="https://github.com/user-attachments/assets/b5e59a54-d948-4f44-b29e-f434b928f72a" width="600">

- 오차율을 상대적으로 표현하여 원본과의 차이를 명확하게 드러내기 위해 수정함.

## 에러율 시각화

<img alt="image" src="https://github.com/user-attachments/assets/ac136eed-bb3f-4bc3-a2df-4351b092827f" width="600">

- **빨간선**: 비정상 데이터, 에러율 범위: 20~50%
- **파란선**: 정상 데이터, 에러율 범위: 0~20%

## 결론

- Autoencoder 모델로 정상 데이터는 높은 정확도로 복원할 수 있음을 확인함.
- 비정상 데이터는 재구성 오차를 통해 명확히 구별할 수 있음을 입증함.
- **향후 계획**:
  - 노이즈 제거 Autoencoder를 추가하여 네트워크 안정성을 더욱 향상할 예정임.

## 참고 자료

1. [음성 오디오를 멜 스펙트로그램으로 매핑하여 생성된 이미지의 유사성을 분석](https://www.mdpi.com/2227-7390/11/3/498)

2. [오디오 신호 처리](https://velog.io/@p2yeong/%EC%98%A4%EB%94%94%EC%98%A4-%EC%B2%98%EB%A6%ACAudio-Processing)

3. [IP 인터컴 시스템](https://www.toa-products.com/international/download/spec/n-8000ex_kr_cb1k.pdf)

4. [딥러닝 음성 처리 파이썬 실습](https://mz-moonzoo.tistory.com/68)

5. [다중 오토인코더를 통한 치매 음성 데이터 이상 신호 감지](https://repository.hanyang.ac.kr/handle/20.500.11754/186520)

6. [오토인코더 이상 탐지 알고리즘의 성능 비교](https://dcoll.ajou.ac.kr/dcollection/srch/srchDetail/000000032583?localeParam=ko)
