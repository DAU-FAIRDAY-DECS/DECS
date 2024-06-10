import wave
import numpy as np

def read_wave_file(filename):
    with wave.open(filename, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        return audio_data, framerate, n_channels, sampwidth

def write_wave_file(filename, audio_data, framerate, n_channels, sampwidth):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(audio_data.tobytes())

def stretch_audio(input_file, output_file):
    input_data, input_rate, input_channels, input_sampwidth = read_wave_file(input_file)
    output_data, output_rate, output_channels, output_sampwidth = read_wave_file(output_file)

    # output.wav 파일의 길이가 input.wav 파일의 길이보다 짧은 경우
    if len(output_data) < len(input_data):
        # 부족한 길이를 계산
        padding_length = len(input_data) - len(output_data)
        # output.wav 파일의 마지막 샘플 값을 가져오고, 파일이 비어 있으면 0으로 초기화
        last_value = output_data[-1] if len(output_data) > 0 else 0
        # 부족한 길이만큼 마지막 샘플 값으로 채움
        padding = np.full(padding_length, last_value, dtype=np.int16)
        # 기존 데이터와 패딩을 결합
        output_data = np.concatenate([output_data, padding])

    # 기존 output.wav 파일 덮어쓰기
    write_wave_file(output_file, output_data, output_rate, output_channels, output_sampwidth)

# 파일 경로 설정
input_file = 'main/voip/wav/input.wav'
output_file = 'main/voip/wav/output.wav'

# input.wav 파일 길이에 맞춰 output.wav 파일 늘리기
stretch_audio(input_file, output_file)