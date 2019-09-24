from scipy import stats
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.signal
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


# Reduces baseline based on all local minima
# Accepts a spectrum's list of intensity values
# Returns baseline-reduced list
def baseline_reduction(list, lam=10**8, p=0.01, niter=10):
    list_length = len(list)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(list_length, list_length - 2))
    w = np.ones(list_length)
    baseline_reduced = []
    for i in range(niter):
        W = sparse.spdiags(w, 0, list_length, list_length)
        Z = W + lam * D.dot(D.transpose())
        baseline = spsolve(Z, w * list)
        w = p * (list > baseline) + (1 - p) * (list < baseline)
    for j in range(list_length):
        if baseline[j] >= 0:
            baseline_reduced.append(list[j] - baseline[j])
        else:
            baseline_reduced.append(list[j] + baseline[j])
    baseline_reduced = np.array(baseline_reduced).clip(min=0)
    return baseline_reduced.tolist()


# Perform smoothing on intensity values
# Accepts a spectrum's list of intensity values
# Returns smoothed list
def smoothing(list):
    # Savitzky-Golay Algorithm
    list_array = np.array(list)
    smoothed_list = scipy.signal.savgol_filter(list_array, 7, 2)
    smoothed_list = smoothed_list.clip(min=0)
    return np.array(smoothed_list).tolist()


# Disregards scale/units of input values
# Accepts a spectrum's list of intensity values
# Returns normalized list
def normalization(list, type):
    # Simple Feature Scaling
    if type == 'sfs':
        list_array = np.array(list)
        list_array[:] = [x / max(list_array) for x in list_array]
        normalized_list = list_array
        return np.array(normalized_list).tolist()

    # Min-Max Normalization
    elif type == 'min_max':
        list_array = np.array(list)
        list_array[:] = [(x - min(list_array)) / (max(list_array) - min(list_array)) for x in list_array]
        normalized_list = list_array
        # normalized_list = preprocessing.normalize([list_array])
        return np.reshape(np.array(normalized_list).tolist(), -1)

    # Z-score Normalization
    elif type == 'z_score':
        list_array = np.array(list)
        normalized_list = stats.zscore(list_array)
        return np.array(normalized_list).tolist()


# Groups data values into bins to lessen dimensions
# Accepts one spectrum input
# Returns reduced spectrum
def data_reduction(spectrum_list, bin=2):
    def mz_binning(mz_list, bin):
        list_array = np.array(mz_list)
        reduced_mz_list = []
        temp = 0
        counter = 0
        for i in range(len(list_array)):
            temp += list_array[i]
            counter += 1
            if counter == len(list_array) and (len(list_array) % bin) != 0:
                reduced_mz_list.append(temp / (len(list_array) % bin))
                temp = 0
            elif counter % bin == 0:
                reduced_mz_list.append(temp / bin)
                temp = 0
        return np.array(reduced_mz_list).tolist()

    def intensity_binning(intensity_list, bin):
        list_array = np.array(intensity_list)
        reduced_i_list = []
        temp = 0
        counter = 0
        for i in range(len(list_array)):
            temp += list_array[i]
            counter += 1
            if counter == len(list_array) and (len(list_array) % bin) != 0:
                reduced_i_list.append(temp / (len(list_array) % bin))
                temp = 0
            elif counter % bin == 0:
                reduced_i_list.append(temp / bin)
                temp = 0
        return np.array(reduced_i_list).tolist()

    reduced_list = [[0 for x in range(2)] for y in range(len(spectrum_list))]
    reduced_list[0] = mz_binning(spectrum_list[0], bin)
    reduced_list[1] = intensity_binning(spectrum_list[1], bin)
    reduced_list[2] = spectrum_list[2]

    return reduced_list


# Checks if particular spectrum is identical to another in data input
# Accepts list of spectra
# Returns peak-aligned list (if duplicate exists)
def peak_alignment(spectrum_list):
    dupli_exists = False
    m, n = 0, 1
    for spectrum in enumerate(spectrum_list):
        for other_spectrum in enumerate(spectrum_list):
            if m == n or len(spectrum_list[m][0]) != len(spectrum_list[n][0]):
                continue
            elif max(spectrum[1]) == max(other_spectrum[1]) and min(spectrum[1]) == min(other_spectrum[1]):
                if max(spectrum[0]) == max(other_spectrum[0]) and min(spectrum[0]) == min(other_spectrum[0]):
                    spectrum_list.remove(other_spectrum)
                    dupli_exists = True
            n += 1
        m += 1
    return dupli_exists


# Retrieves dataset and performs pre-processing after parsing
# Returns preprocessed spectrum list
def get_preprocessed_data(spectrum_list, parameters):
    i = 0
    while i < len(spectrum_list):
        spectrum = spectrum_list[i]

        # Pre-processing Inputted data
        if 'bl_reduction' in parameters and len(spectrum[1]) > 1:
            spectrum[1] = baseline_reduction(spectrum[1])
        if 'smoothing' in parameters and len(spectrum[1]) >= 7:
            spectrum[1] = smoothing(spectrum[1])
        if 'sfs' in parameters and spectrum[1]:
            spectrum[1] = normalization(spectrum[1], 'sfs')
        if 'min_max' in parameters and spectrum[1]:
            spectrum[1] = normalization(spectrum[1], 'min_max')
        if 'z_score' in parameters and spectrum[1]:
            spectrum[1] = normalization(spectrum[1], 'z_score')
        if 'data_reduction' in parameters and spectrum[1]:
            bins = False
            if 'number_of_bins' in parameters:
                bins = int(parameters[parameters.index('number_of_bins') + 1])
            if bins:
                spectrum = data_reduction(spectrum, bins)
            else:
                spectrum = data_reduction(spectrum)
        spectrum_list[i] = spectrum
        i += 1
    used_pa, dupli_exists = False, False
    if 'peak_alignment' in parameters:
        used_pa = True
        dupli_exists = peak_alignment(spectrum_list)

    return spectrum_list, used_pa, dupli_exists
