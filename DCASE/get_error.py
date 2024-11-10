import csv
import pandas as pd

# 데이터를 파일로 저장한 파일 경로 지정
file_path = "normal_evg.csv" 

# CSV 파일에서 데이터 읽기
data = pd.read_csv(file_path, header=None, names=['file_name', 'value'])

# 평균 값 계산
average_normal_error = data['value'].mean()

print("평균값:", average_normal_error)

# CSV 파일 경로
input_csv_path = "result\\anomaly_score_test.csv"  # 재구성 오차 CSV 파일 경로
output_csv_path = "result\error_rates.csv"           # 결과 저장 파일 경로

# CSV 파일 읽기 및 에러율 계산
error_rates = []
with open(input_csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        file_name = row[0]
        reconstruction_error = float(row[1])

        # 에러율 계산
        error_rate =  (1 - (average_normal_error / reconstruction_error)) * 100
        error_rates.append([file_name, reconstruction_error, error_rate])

# 계산 결과를 새로운 CSV 파일로 저장
with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(error_rates)

print(f"TEST 데이터의 에러율 계산이 완료되었습니다. 결과가 {output_csv_path}에 저장되었습니다.")
