import numpy as np
import librosa
import os

def save_feature_to_file(feature, filename):
    np.savetxt(filename, np.round(feature, 3), fmt='%.3f')

def extract_audio_features_per_frame(file_path, output_dir, frame_size=1024, hop_length=512):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lists to store features for each frame
    energy_list = []
    zero_crossing_rate_list = []
    fundamental_frequency_list = []
    spectral_centroid_list = []
    spectral_bandwidth_list = []
    spectral_rolloff_list = []

    # Process each frame
    for i in range(0, len(y) - frame_size + 1, hop_length):
        frame = y[i:i + frame_size]

        # Time-domain features
        energy = np.sum(np.square(frame))
        energy_list.append(energy)

        zero_crossings = librosa.feature.zero_crossing_rate(frame, frame_length=frame_size, hop_length=hop_length)
        zero_crossing_rate_list.append(zero_crossings.mean())

        f0, voiced_flag, voiced_probs = librosa.pyin(frame, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
        f0 = f0[~np.isnan(f0)]
        fundamental_freq = f0.mean() if len(f0) > 0 else 0
        fundamental_frequency_list.append(fundamental_freq)

        # Frequency-domain features
        spectral_centroid = librosa.feature.spectral_centroid(y=frame, sr=sr)
        spectral_centroid_list.append(spectral_centroid.mean())

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=frame, sr=sr)
        spectral_bandwidth_list.append(spectral_bandwidth.mean())

        spectral_rolloff = librosa.feature.spectral_rolloff(y=frame, sr=sr, roll_percent=0.85)
        spectral_rolloff_list.append(spectral_rolloff.mean())

    # Save all extracted features to respective files with rounding and formatting
    save_feature_to_file(energy_list, os.path.join(output_dir, 'energy_per_frame.txt'))
    save_feature_to_file(zero_crossing_rate_list, os.path.join(output_dir, 'zero_crossing_rate_per_frame.txt'))
    save_feature_to_file(fundamental_frequency_list, os.path.join(output_dir, 'fundamental_frequency_per_frame.txt'))
    save_feature_to_file(spectral_centroid_list, os.path.join(output_dir, 'spectral_centroid_per_frame.txt'))
    save_feature_to_file(spectral_bandwidth_list, os.path.join(output_dir, 'spectral_bandwidth_per_frame.txt'))
    save_feature_to_file(spectral_rolloff_list, os.path.join(output_dir, 'spectral_rolloff_per_frame.txt'))

# Example usage
input_wav_file = '/Volumes/JMJ_SSD/wavFeatures/noiseMixedWav/output0001.wav'  # Replace with your .wav file path
output_directory = 'features_output_per_frame'
extract_audio_features_per_frame(input_wav_file, output_directory)
