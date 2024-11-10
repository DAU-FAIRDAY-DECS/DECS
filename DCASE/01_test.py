########################################################################
# import default libraries
########################################################################
import os
import csv
import sys
import gc
########################################################################


########################################################################
# import additional libraries
########################################################################
import numpy as np
import scipy.stats
# from import
from tqdm import tqdm
from sklearn import metrics
try:
    from sklearn.externals import joblib
except:
    import joblib

# original lib
import common as com
import keras_model

########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################


########################################################################
# output csv file
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


########################################################################


########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":

    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    performance_over_all = []

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx+1, total=len(dirs)))

        print("============== MODEL LOAD ==============")
        # load model file
        model_file = "{model}/model_100_64.keras".format(model=param["model_directory"])
        if not os.path.exists(model_file):
            com.logger.error("model not found ")
            sys.exit(-1)
        model = keras_model.load_model(model_file)
        model.summary()

        # load anomaly score distribution for determining threshold
        score_distr_file_path = "{model}/score_distr.pkl".format(model=param["model_directory"])
        shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)

        # determine threshold for decision
        #decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)
        decision_threshold = 40
        print("Decision Threshold:", decision_threshold)


        # results
        csv_lines.append(["precision", "recall", "F1 score"])
        performance = []

        dir_name = "test"
    
        # load test file
        files, y_true = com.file_list_generator(target_dir=target_dir,
                                                section_name="*",
                                                dir_name=dir_name)

        # setup anomaly score file path
        anomaly_score_csv = "{result}/anomaly_score_{dir_name}.csv".format(result=param["result_directory"], dir_name=dir_name)
        anomaly_score_list = []

        # setup decision result file path
        decision_result_csv = "{result}/decision_result_{dir_name}.csv".format(result=param["result_directory"], dir_name=dir_name)
        decision_result_list = []

        print("\n============== BEGIN TEST FOR A SECTION ==============")
        y_pred = [0. for k in files]
        for file_idx, file_path in tqdm(enumerate(files), total=len(files)):
            try:
                data = com.file_to_vectors(file_path,
                                                n_mels=param["feature"]["n_mels"],
                                                n_frames=param["feature"]["n_frames"],
                                                n_fft=param["feature"]["n_fft"],
                                                hop_length=param["feature"]["hop_length"],
                                                power=param["feature"]["power"])
            except:
                com.logger.error("File broken!!: {}".format(file_path))

            y_pred[file_idx] = np.mean(np.square(data - model.predict(data)))
            
            # store anomaly scores
            anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])

            # store decision results
            if y_pred[file_idx] > decision_threshold:
                decision_result_list.append([os.path.basename(file_path), 1])
            else:
                decision_result_list.append([os.path.basename(file_path), 0])

        # output anomaly scores
        save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
        com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

        # output decision results
        save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
        com.logger.info("decision result ->  {}".format(decision_result_csv))

        # extract scores for calculation of AUC (source) and AUC (target)
        y_true_s_auc = [y_true[idx] for idx in range(len(y_true)) if  y_true[idx]==1]
        y_pred_s_auc = [y_pred[idx] for idx in range(len(y_true)) if  y_true[idx]==1]

        # extract scores for calculation of precision, recall, F1 score for each domain
        y_true_s = [y_true[idx] for idx in range(len(y_true))]
        y_pred_s = [y_pred[idx] for idx in range(len(y_true))]

        #p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
        tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(y_true_s, [1 if x > decision_threshold else 0 for x in y_pred_s]).ravel()
        prec_s = tp_s / np.maximum(tp_s + fp_s, sys.float_info.epsilon)
        recall_s = tp_s / np.maximum(tp_s + fn_s, sys.float_info.epsilon)
        f1_s = 2.0 * prec_s * recall_s / np.maximum(prec_s + recall_s, sys.float_info.epsilon)

        csv_lines.append([prec_s, recall_s, f1_s])

        performance.append([prec_s, recall_s, f1_s])
        performance_over_all.append([prec_s, recall_s, f1_s])

        # com.logger.info("AUC (source) : {}".format(auc_s))
        # com.logger.info("pAUC : {}".format(p_auc))
        com.logger.info("precision (source) : {}".format(prec_s))
        com.logger.info("recall (source) : {}".format(recall_s))
        com.logger.info("F1 score (source) : {}".format(f1_s))

        print("\n============ END OF TEST FOR A SECTION ============")

        del data
        del model
        keras_model.clear_session()
        gc.collect()

    csv_lines.append(["Decision Threshold:" , decision_threshold])
        
    # output results
    result_path = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    com.logger.info("results -> {}".format(result_path))
    save_csv(save_file_path=result_path, save_data=csv_lines)
