from sklearn.model_selection import train_test_split
from joblib import load
import statistics


def predict(spectrum_list, preproc_param):
    # Set data labels
    labels = set_labels(spectrum_list, preproc_param)

    # Get mean of each spectrum element
    for spectrum in spectrum_list:
        total_mz = 0
        print(spectrum[0])
        for mz in spectrum[0]:
            total_mz += mz
        spectrum[0] = total_mz / len(spectrum[0])

        total_intensity= 0
        for i in spectrum[1]:
            total_intensity += mz
        spectrum[1] = total_intensity / len(spectrum[1])

        total_rt = 0
        for rt in spectrum[2]:
            total_rt += rt
        spectrum[2] = rt

    # Set data
    x = spectrum_list

    # Split data to train and test on 80-20 ratio
    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=0)

    # Load RBF-SVM classifier
    clf = load_model(preproc_param)

    # Make predictions on unseen test data
    clf_predictions = clf.predict(x_test)

    # Compute the probability values for the predictions
    clf_probabilities = clf.predict_proba(x_test)[:, 1] if type else clf.predict_proba(x_test)[:, 0]
    final_probability = statistics.mean(clf_probabilities)

    # Set final classification result
    final_prediction = 'Positive' if statistics.mode(clf_predictions) == 1 else 'Negative'

    return final_prediction, final_probability


def set_labels(list, param):
    labels = []
    i = 0

    # Adjust label intervals to each unique SVM model
    if 'bl_reduction' in param and 'smoothing' in param and len(list) == 2:
        format1 = 6
        format2 = 6
    elif 'bl_reduction' in param and 'smoothing' in param and 'sfs' in param and len(param) == 3:
        format1 = 7
        format2 = 7
    elif 'bl_reduction' in param and 'smoothing' in param and 'min_max' in param and len(param) == 3:
        format1 = 6
        format2 = 6
    elif 'bl_reduction' in param and 'smoothing' in param and ('sfs' in param or 'min_max' in param or 'z_score' in param) and 'data_reduction' in param and len(param) == 4:
        format1 = 6
        format2 = 9
    elif 'bl_reduction' in list and 'smoothing' in param and ('sfs' in param or 'min_max' in param or 'z_score' in param) and 'data_reduction' in param and len(param) == 4:
        format1 = 10
        format2 = 10
    else:
        format1 = 4
        format2 = 4

    while i < len(list) / 2:
        if i % format1 == 0:
            labels.append(1)
        else:
            labels.append(0)
        i += 1
    j = 0
    while j < len(list) / 2:
        if j % format2 == 0:
            labels.append(0)
        else:
            labels.append(1)
        j += 1
    return labels


def load_model(pp_parameters):
    # Load the classifier model corresponding to the pre-processing steps
    if 'bl_reduction' in pp_parameters and 'smoothing' in pp_parameters and len(pp_parameters) == 2:
        with open('svm_models/svm_model_one.joblib', 'rb') as fo:
            clf = load(fo)
    elif 'bl_reduction' in pp_parameters and 'smoothing' in pp_parameters and 'sfs' in pp_parameters and len(pp_parameters) == 3:
        with open('svm_models/svm_model_two.joblib', 'rb') as fo:
            clf = load(fo)
    elif 'bl_reduction' in pp_parameters and 'smoothing' in pp_parameters and 'min_max' in pp_parameters and len(pp_parameters) == 3:
        with open('svm_models/svm_model_three.joblib', 'rb') as fo:
            clf = load(fo)
    elif 'bl_reduction' in pp_parameters and 'smoothing' in pp_parameters and 'z_score' in pp_parameters and len(pp_parameters) == 3:
        with open('svm_models/svm_model_four.joblib', 'rb') as fo:
            clf = load(fo)
    elif 'bl_reduction' in pp_parameters and 'smoothing' in pp_parameters and ('sfs' in pp_parameters or 'min_max' in pp_parameters or 'z_score' in pp_parameters) and 'data_reduction' in pp_parameters  and len(pp_parameters) == 4:
        with open('svm_models/svm_model_five.joblib', 'rb') as fo:
            clf = load(fo)
    else:
        with open('svm_models/svm_model_six.joblib', 'rb') as fo:
            clf = load(fo)
    return clf


def none():
    return None
