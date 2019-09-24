# Main file used for training, testing, and cross validation
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from joblib import dump, load
import statistics
import data_parser
import preprocessing
import time
import numpy as np


# Create model using spectral data
def train_test_model(spectrum_list, preproc_param):
    # Set data labels
    labels = set_labels(spectrum_list, preproc_param)
    target_names = ['0', '1']

    # Get mean of each spectrum element
    for spectrum in spectrum_list:
        total_mz = 0
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

    # Create an RBF-SVM classifier
    clf = svm.SVC(C=1, kernel='rbf', gamma=0.001, probability=True)
    # clf = load_model(preproc_param)

    # Train classifier
    clf.fit(x_train, y_train)

    # Make predictions on unseen test data
    clf_predictions = clf.predict(x_test)

    # Compute the probability values for the predictions
    clf_probabilities = clf.predict_proba(x_test)[:, 1] if type else clf.predict_proba(x_test)[:, 0]
    final_probability = statistics.mean(clf_probabilities)

    # Set final classification result
    final_prediction = 'Positive' if statistics.mode(clf_predictions) == 1 else 'Negative'

    # Persist model depending on the processing methods
    save_model(clf, preproc_param)

    print('Pancreatic Cancer: {}'.format(final_prediction))
    print('Confidence Level: {}%'.format(round((final_probability * 100), 3)))
    print('Accuracy: {}%'.format(round((clf.score(x_test, y_test) * 100), 3)))

    # Get scores
    print(classification_report(y_test, clf_predictions, target_names=target_names))
    print('Accuracy Score {}'.format(round((accuracy_score(y_test, clf_predictions)), 3)))
    print('Balanced Accuracy Score {}'.format(round((balanced_accuracy_score(y_test, clf_predictions)), 3)))
    print('Precision Score {}'.format(round((precision_score(y_test, clf_predictions)), 3)))
    print('Recall Score {}'.format(round((recall_score(y_test, clf_predictions)), 3)))
    print('F1-Score {}'.format(round((f1_score(y_test, clf_predictions)), 3)))
    return None


# Set data labels
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


# Cross Validation
def cross_validate(total_data,):
    # Set data labels
    labels = set_labels(total_data)

    # Define hyperparameter grid
    param_grid = {'C': [1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.1, 0.01, 0.001, 0.0001]}

    # Create GridSearchCV
    grid_search = GridSearchCV(svm.SVC(class_weight='balanced'), param_grid)

    # Record time
    start = time.time()

    # Start GridSearchCV
    grid_search = grid_search.fit(total_data, labels)
    print("Best estimator found by grid search:")
    print(grid_search.best_estimator_)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time.time() - start, len(grid_search.cv_results_['params'])))

    # Split data to train and test on 80-20 ratio
    x_train, x_test, y_train, y_test = train_test_split(total_data, labels, test_size=0.2, random_state=0)
    predictions = grid_search.predict(x_test)
    print('Accuracy: {}%'.format(round((clf.score(x_test, y_test) * 100), 3)))
    print(classification_report(y_test, predictions))

    # Show GridSearchCV Results
    report(grid_search.cv_results_)
    return None


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# Persist Model
def save_model(classifier, pp_parameters):
    # Persist the classifier object to the model corresponding to the pre-processing steps
    if 'bl_reduction' in pp_parameters and 'smoothing' in pp_parameters and len(pp_parameters) == 2:
        with open('svm_models/svm_model_one.joblib', 'wb') as fo:
            dump(classifier, fo)
    elif 'bl_reduction' in pp_parameters and 'smoothing' in pp_parameters and 'sfs' in pp_parameters and len(pp_parameters) == 3:
        with open('svm_models/svm_model_two.joblib', 'wb') as fo:
            dump(classifier, fo)
    elif 'bl_reduction' in pp_parameters and 'smoothing' in pp_parameters and 'min_max' in pp_parameters and len(pp_parameters) == 3:
        with open('svm_models/svm_model_three.joblib', 'wb') as fo:
            dump(classifier, fo)
    elif 'bl_reduction' in pp_parameters and 'smoothing' in pp_parameters and 'z_score' in pp_parameters and len(pp_parameters) == 3:
        with open('svm_models/svm_model_four.joblib', 'wb') as fo:
            dump(classifier, fo)
    elif 'bl_reduction' in pp_parameters and 'smoothing' in pp_parameters and ('sfs' in pp_parameters or 'min_max' in pp_parameters or 'z_score' in pp_parameters) and 'data_reduction' in pp_parameters and len(pp_parameters) == 4:
        with open('svm_models/svm_model_five.joblib', 'wb') as fo:
            dump(classifier, fo)
    else:
        with open('svm_models/svm_model_six.joblib', 'wb') as fo:
            dump(classifier, fo)


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
        with open('svm_model_six.joblib', 'rb') as fo:
            clf = load(fo)
    return clf


def main():
    # Class Negative Data
    d1 = data_parser.parse('Datasets/Healthy Controls/MS_A_1.mzml')
    d2 = data_parser.parse('Datasets/Healthy Controls/MS_A_2.mzml')
    d3 = data_parser.parse('Datasets/Healthy Controls/MS_A_3.mzml')
    d4 = data_parser.parse('Datasets/Healthy Controls/MS_A_4.mzml')
    d5 = data_parser.parse('Datasets/Healthy Controls/MS_A_5.mzml')
    d6 = data_parser.parse('Datasets/Healthy Controls/MS_A_6.mzml')
    d7 = data_parser.parse('Datasets/Healthy Controls/MS_A_7.mzml')

    # Class Positive Data
    d8 = data_parser.parse('Datasets/PC Diagnosed/MS_B_1.mzml')
    d9 = data_parser.parse('Datasets/PC Diagnosed/MS_B_2.mzml')
    d10 = data_parser.parse('Datasets/PC Diagnosed/MS_B_3.mzml')
    d11 = data_parser.parse('Datasets/PC Diagnosed/MS_B_4.mzml')
    d12 = data_parser.parse('Datasets/PC Diagnosed/MS_B_5.mzml')
    d13 = data_parser.parse('Datasets/PC Diagnosed/MS_B_6.mzml')
    d14 = data_parser.parse('Datasets/PC Diagnosed/MS_B_7.mzml')

    full_data = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9 + d10 + d11 + d12 + d13 + d14
    param = []
    data = preprocessing.get_preprocessed_data(full_data, param)
    # train_test_model(data, param)
    cross_validate(data, param)


if __name__ == "__main__":
    main()
