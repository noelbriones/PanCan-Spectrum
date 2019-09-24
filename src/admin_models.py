from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from joblib import dump


def train_test_model(spectrum_list):
    # Set data labels
    labels = set_labels(spectrum_list)
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

    # Train classifier
    clf.fit(x_train, y_train)

    # Make predictions on unseen test data
    clf_predictions = clf.predict(x_test)

    print('Accuracy: {}%'.format(round((clf.score(x_test, y_test) * 100), 3)))

    # Get scores
    accuracy = round(accuracy_score(y_test, clf_predictions), 3)
    precision = round(precision_score(y_test, clf_predictions), 3)
    recall = round(recall_score(y_test, clf_predictions), 3)
    f1 = round(f1_score(y_test, clf_predictions), 3)

    return clf, accuracy, precision, recall, f1


# Set data labels
def set_labels(list):
    labels = []
    i = 0

    # Ensure two classes are produced for the classifier
    while i < len(list) / 2:
        if i % 10 == 0:
            labels.append(1)
        else:
            labels.append(0)
        i += 1
    j = 0
    while j < len(list) / 2:
        if j % 10 == 0:
            labels.append(0)
        else:
            labels.append(1)
        j += 1
    return labels


def save_model(classifier, location, name):
    # Persist the classifier object to the model corresponding to the pre-processing steps
    with open(location + '/' + name, 'wb') as fo:
        dump(classifier, fo)
