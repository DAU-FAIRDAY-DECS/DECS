########################################################################
# import default libraries
########################################################################
import os
import gc
########################################################################


########################################################################
# import additional libraries
########################################################################
import numpy as np
import scipy.stats

from tqdm import tqdm

import joblib

import common as com
import keras_model
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################

########################################################################
# get data from the list for file paths
########################################################################
def file_list_to_data(file_list,
                      msg="calc...",
                      n_mels=64,
                      n_frames=5,
                      n_hop_frames=1,
                      n_fft=1024,
                      hop_length=512,
                      power=2.0):
   
    # calculate the number of dimensions
    dims = n_mels * n_frames
    data_list = []  # 리스트를 사용해 데이터를 동적으로 할당

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vectors = com.file_to_vectors(file_list[idx],
                                      n_mels=n_mels,
                                      n_frames=n_frames,
                                      n_fft=n_fft,
                                      hop_length=hop_length,
                                      power=power)
        vectors = vectors[::n_hop_frames, :]
        data_list.append(vectors)  # 데이터를 리스트에 추가

    # 리스트에 있는 모든 벡터를 하나의 numpy 배열로 결합
    data = np.vstack(data_list)
    
    return data


########################################################################


########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":

    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # load base_directory list
    dirs = com.select_dirs(param=param)
    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        model_file_path = "{model}/model_100_64.keras".format(model=param["model_directory"])

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue
        
        # pickle file for storing anomaly score distribution
        score_distr_file_path = "{model}/score_distr.pkl".format(model=param["model_directory"])

        # generate dataset
        print("============== DATASET_GENERATOR ==============")

        # get file list for all sections
        # all values of y_true are zero in training
        files, y_true = com.file_list_generator(target_dir=target_dir,
                                                section_name="*",
                                                dir_name="train")

        data = file_list_to_data(files,
                                msg="generate train_dataset",
                                n_mels=param["feature"]["n_mels"],
                                n_frames=param["feature"]["n_frames"],
                                n_hop_frames=param["feature"]["n_hop_frames"],
                                n_fft=param["feature"]["n_fft"],
                                hop_length=param["feature"]["hop_length"],
                                power=param["feature"]["power"])

        # number of vectors for each wave file
        n_vectors_ea_file = int(data.shape[0] / len(files))

        # train model
        print("============== MODEL TRAINING ==============")
        model = keras_model.get_model(param["feature"]["n_mels"] * param["feature"]["n_frames"],
                                    param["fit"]["lr"])

        model.summary()

        history = model.fit(x=data,
                            y=data,
                            epochs=param["fit"]["epochs"],
                            batch_size=param["fit"]["batch_size"],
                            shuffle=param["fit"]["shuffle"],
                            validation_split=param["fit"]["validation_split"],
                            verbose=param["fit"]["verbose"])

        # calculate y_pred for fitting anomaly score distribution
        y_pred = []
        start_idx = 0
        for file_idx in range(len(files)):
                y_pred.append(np.mean(np.square(data[start_idx : start_idx + n_vectors_ea_file, :] 
                                    - model.predict(data[start_idx : start_idx + n_vectors_ea_file, :]))))
                start_idx += n_vectors_ea_file

        # fit anomaly score distribution
        shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred)
        gamma_params = [shape_hat, loc_hat, scale_hat]
        joblib.dump(gamma_params, score_distr_file_path)
        
        model.save(model_file_path)
        com.logger.info("save_model -> {}".format(model_file_path))
        print("============== END TRAINING ==============")

        del data
        del model
        keras_model.clear_session()
        gc.collect()