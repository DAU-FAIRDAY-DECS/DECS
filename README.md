# DECS
### 디지털화된 오디오를 네트워크로 전송시 에러율 확인 소프트웨어
>**D**etection **E**rror in **C**ommunication **S**ystem
- 소속 : 동아대학교 컴퓨터공학과<br/> 
- 개발기간 : 2024.03 ~ ing<br/> 

### 팀원
| 이름 | 역할 |
|-----------|-----------|
| 정성윤(팀장) | Autoencoder 이상 감지 모델 구성 및 학습 |
| 강태원 | Autoencoder 이상 감지 모델 구성 및 학습 |
| 박유진 | UDP 기반 VoIP 프로토콜 통신 및 시각화 |
| 전민재 | UDP 기반 VoIP 프로토콜 통신 및 시각화 |

# 프로젝트 소개
디지털 오디오 네트워크 전송 에러율 확인 소프트웨어는 네트워크를 통해 전송된 오디오 데이터의 에러율을 감지하고 분석하여, 네트워크 전송의 안정성을 향상시키는 것을 목표로 합니다.

### 예상 성과
- **에러율 감지 및 분석 기능 제공**
  - 송수신된 오디오를 비교하여 에러율을 신속하게 감지하고 통계적 분석을 제공하여 네트워크 전송의 안정성을 향상시킵니다.

### 개발 배경 및 필요성
- **인터컴 시스템의 IP화 요구**
  - 기존의 함정 내 통신 장비를 IP 방식으로 연동하기 위한 요구가 증가하고 있습니다.
- **다양한 송수신 장비와 오디오 품질 유지**
  - 다수의 장비로부터 동시에 송수신되는 오디오를 네트워크를 통해 전송하고, 이를 수신국에서 복호화하여 일정 수준 이상의 오디오 품질을 보장해야 합니다.
- **네트워크 중간 믹서의 성능 검증**
  - 중계되는 오디오의 품질을 확인하고 믹서의 성능을 검증하기 위해 에러율 확인 소프트웨어가 필요합니다.

# 개발

## 아날로그 인터컴 디지털화
아날로그 인터컴 시스템을 사용하여 음성 통신을 디지털 형식으로 전환합니다. 아날로그 오디오 신호를 8kHz 샘플링 레이트와 16비트 해상도로 디지털 변환합니다.

## 음성 오디오 디지털화
- **오디오 포맷**: 16비트 사용
- **샘플링 레이트**: 8000Hz
- **채널 수**: 단일 오디오 채널
- **버퍼 크기**: 오디오 프레임 1024개

## VoIP 프로토콜 통신
VoIP 프로토콜을 통해 IP 네트워크 상태에서 패킷으로 전송합니다. 네트워크 전송 과정에서 발생할 수 있는 패킷 손실, 지연, 순서 등의 이슈를 처리합니다.

## 송신자 서버
![image](https://github.com/paul0817/Markdown/assets/100745610/12ba43e3-72cb-4858-bae6-063d2c09684b)
1. PyAudio 초기화 및 오디오 입력 스트림 개방
2. UDP 소켓 생성
3. PCM 데이터를 ADPCM 데이터로 압축
4. 패킷 이슈 시뮬레이션
5. UDP 소켓을 통한 데이터 전송

## 수신자 서버
![image](https://github.com/paul0817/Markdown/assets/100745610/e5728c5d-5071-4503-93b9-65c735a96cc9)
1. PyAudio 초기화 및 오디오 출력 스트림 개방
2. UDP 소켓 생성
3. 수신된 데이터를 ADPCM 데이터로 재조립
4. 패킷 이슈 시뮬레이션
5. ADPCM 데이터를 PCM 데이터로 압축 해제

## ADPCM 압축
ADPCM 코덱을 사용하여 오디오 데이터를 압축합니다. 이는 원본 음성의 품질을 유지하면서 높은 압축 비율을 제공합니다.

## 통신 결과
 - 패킷 이슈 시뮬레이션을 적용하지 않은 통신 웨이브폼으로, 약간의 패킷 지연은 발생하나 로스가 거의 발생하지 않음

![image](https://github.com/paul0817/Markdown/assets/100745610/b8283a04-0ae0-4ffe-b66a-5dc0e8aaa16a)
<br/><br/>
 - 정상과 비정장 데이터의 구별을 극대화하기 위해 패킷 이슈 시뮬레이션을 삽입한 통신 웨이브폼임
   
![image](https://github.com/paul0817/Markdown/assets/100745610/6e74fb40-6240-47a3-908c-f20b494662e6)


## Anomaly Detection
기계 학습 모델을 사용하여 통신 중 발생하는 이상 패킷을 실시간으로 감지하고, 감지된 이상 현상을 분석하여 통신 에러율을 계산합니다. 오토인코더(Autoencoder) 기반의 비지도 학습을 사용하여 정상 및 비정상 데이터를 구분합니다.
- **Autoencoder**: 입력 데이터를 압축한 후 복원하는 신경망
  - **Encoder**: Input 데이터를 압축
  - **Decoder**: 압축된 데이터를 복원
  - **Latent Vector**: 압축 과정에서 추출한 의미 있는 데이터
 
## 학습 결과
![image](https://github.com/paul0817/Markdown/assets/100745610/260fb073-b473-41c8-b796-6d5a11844964)
> 재구성 오류 분포 그래프
- 정상 데이터의 재구성 오류가 매우 낮음
  - 모델이 정상 데이터를 거의 완벽하게 재구성할 수 있으며, 모델이 정상 패턴을 잘 학습했음을 의미
- 비정상 데이터의 재구성 오류가 높음
  - 비정상 데이터의 재구성 오류가 정상 데이터보다 높게 분포하고 있으며, 모델이 비정상 데이터를 잘 재구성하지 못함을 의미
<br/>


![image](https://github.com/paul0817/Markdown/assets/100745610/70470a42-3d1f-49e4-a026-f29af528a0f1)
> 임계값 범위 탐색 그래프
- 정상 데이터의 재구성 오류가 낮음
  - 모델이 정상 데이터를 거의 완벽하게 재구성할 수 있음
- 비정상 데이터의 재구성 오류가 높음
  - 모델이 비정상 데이터를 재구성하는 데 어려움을 겪고 있으며, 정상 데이터와 비정상 데이터를 구별할 수 있는 중요한 지표가 됨

## 모델 성능 평가
![image](https://github.com/paul0817/Markdown/assets/100745610/bc9c0c5d-f48b-4ede-a423-18dfe23d9776)
> 혼동행렬

### 정확도 (Accuracy)
정확도는 모델이 정확하게 예측한 샘플의 비율을 나타내는 지표입니다.

### 정밀도 (Precision)
정밀도는 모델이 양성으로 예측한 샘플 중 실제로 양성인 샘플의 비율을 나타내는 지표입니다.

### 재현률 (Recall)
재현률은 실제 양성인 샘플 중에서 모델이 올바르게 양성으로 예측한 비율을 나타내는 지표입니다.

|                 | Predicted Positive | Predicted Negative |
|-----------------|--------------------|--------------------|
| Actual Positive | TP = 191           | FN = 9             |
| Actual Negative | FP = 23            | TN = 3             |
- **정확도**: 85%
- **정밀도**: 89%
- **재현율**: 95%
- **혼동 행렬**: 정상 데이터를 거의 완벽하게 재구성하지만 비정상 데이터를 구분하는 데 어려움을 겪음

## References

1. [음성 오디오를 멜 스펙트로그램으로 매핑하여 생성된 이미지의 유사성을 분석](https://www.mdpi.com/2227-7390/11/3/498)

2. [오디오 신호 처리](https://velog.io/@p2yeong/%EC%98%A4%EB%94%94%EC%98%A4-%EC%B2%98%EB%A6%ACAudio-Processing)

3. [IP 인터컴 시스템](https://www.toa-products.com/international/download/spec/n-8000ex_kr_cb1k.pdf)

4. [딥러닝 음성 처리 파이썬 실습](https://mz-moonzoo.tistory.com/68)

5. [다중 오토인코더를 통한 치매 음성 데이터 이상 신호 감지](https://repository.hanyang.ac.kr/handle/20.500.11754/186520)

6. [오토인코더 이상 탐지 알고리즘의 성능 비교](https://dcoll.ajou.ac.kr/dcollection/srch/srchDetail/000000032583?localeParam=ko)

