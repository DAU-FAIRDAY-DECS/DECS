import os
import sys

import numpy as np
import scipy.stats

import joblib
import common as com
import keras_model

param = com.yaml_load() 

#Calculate reconstruction error for a single wav file
def calculate_reconstruction_error(file_path, model, decision_threshold):
    try:
        data = com.file_to_vectors(
            file_name=file_path,
            n_mels=param["feature"]["n_mels"],
            n_frames=param["feature"]["n_frames"],
            n_fft=param["feature"]["n_fft"],
            hop_length=param["feature"]["hop_length"],
            power=param["feature"]["power"]
        )

        reconstructed = model.predict(data)

        mse = np.mean(np.square(data - reconstructed))

        print(f"Reconstruction Error (MSE): {mse:.4f}")
        if mse > decision_threshold:
            print(f"Anomaly Detected!")
        else:
            print(f"Normal!")

        return mse

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

if __name__ == "__main__":

    file_path = "dev_data\\human\\test\\0001-loss.wav"


    model_file = f"{param['model_directory']}/model_human_100_64.keras"
    score_distr_file = f"{param['model_directory']}/score_distr_human.pkl"
    
    # Load model
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found!")
        sys.exit(-1)
    
    model = keras_model.load_model(model_file)
    model.summary()

    shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file)
    decision_threshold = 40 #scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)
    print(f"Loaded Decision Threshold: {decision_threshold}")

    calculate_reconstruction_error(file_path, model, decision_threshold)
