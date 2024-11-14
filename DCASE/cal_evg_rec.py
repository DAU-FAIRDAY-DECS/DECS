import pandas as pd

# 데이터를 파일로 저장했을 경우 파일 경로 지정
file_path = "normal_evg.csv"  # 예: "data.csv"

# CSV 파일에서 데이터 읽기 (파일이 아닌 경우 직접 리스트로 생성해도 됩니다)
data = pd.read_csv(file_path, header=None, names=['file_name', 'value'])

# 평균 값 계산
average_value = data['value'].mean()

print("평균값:", average_value)
    