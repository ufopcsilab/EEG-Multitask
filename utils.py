import numpy as np
import pyedflib
from scipy.signal import firwin, filtfilt
from keras.utils.np_utils import to_categorical


def read_edf(file_path, channels=None):
    reader = pyedflib.EdfReader(file_path)

    if channels:
        signals = []
        signal_labels = reader.getSignalLabels()
        for c in channels:
            index = signal_labels.index(c)
            signals.append(reader.readSignal(index))
        signals = np.array(signals)
    else:
        n = reader.signals_in_file
        signals = np.zeros((n, reader.getNSamples()[0]))
        for i in np.arange(n):
            signals[i, :] = reader.readSignal(i)

    reader._close()
    del reader
    return signals


def filter_signal(signal,fs,lowcut,highcut,order):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    fir_coeff = firwin(order+1,[low,high], pass_zero=False)
    return filtfilt(fir_coeff, 1.0, signal)


def normalize(data):
    data = data - np.mean(data,axis=0)
    data = data + np.abs(np.amin(data,axis=0))
    data = data/np.std(data,axis=0)
    data = data/np.amax(data,axis=0)
    return data


def count_total_samples(subs, train_tasks, step):
    samples_matrix = np.zeros((subs, len(train_tasks)),dtype=int)

    for task in range(len(train_tasks)):
        for sub in range(subs):
            data = read_edf(r'/media/share/marianamota/all_data/S%03d%s.edf' % (sub+1,train_tasks[task]))
            #data = read_edf(r'C:\Users\User\Desktop\all_data\S%03d%s.edf' % (sub+1,train_tasks[task]))
            size = len(data.T)
            samples = np.ceil((size-1920)/step)
            samples_matrix[sub,task] = samples

    del data, size, samples
    total = int(np.sum(samples_matrix))

    return total, samples_matrix


def generate_ids(subs, train_tasks):
    pre_ids = []
    for row in range(subs):
        for col in train_tasks:
            aux = ('S%03d%s'%(row+1,col) , str(row))
            pre_ids.append(aux)

    ids = dict(pre_ids)
    return ids


def generate_samples(train_tasks,n_classes, step, samples_matrix):
    train_samples = []
    val_samples = []
    count_task = 0
    for task in train_tasks:
       for subject in range(n_classes):
            start = 0
            train = np.floor(0.9 * samples_matrix[subject, count_task])
            for quant in range(samples_matrix[subject, count_task]):
                if quant < train:
                    train_samples.append('%s-%d-%d' % (task,subject,start))
                else:
                    val_samples.append('%s-%d-%d' % (task, subject, start))
                start = start + step
       count_task = count_task + 1

    return train_samples, val_samples


def load_test_samples(classes, test_task, channels=None):
    x_test = []
    labels_test = []

    for i in range(classes):
        data_test = read_edf(r'G:\Meu Drive\EEG Motor MovementImagery Dataset\S%03d\S%03d' % (i + 1, i + 1) + test_task + '.edf', channels=channels)
        data_test = filter_signal(data_test, fs=160, lowcut=30, highcut=50, order=12)
        data_test = data_test.T
        data_test = normalize(data_test)

        samples_test = int(len(data_test)/1920)

        for j in range(samples_test):
            x_test.append(data_test[1920 * j:(1920 * j + 1920), :])
            labels_test.append(i)

    labels_test = np.array(labels_test)
    x_test = np.array(x_test)
    y_test = to_categorical(labels_test)

    return x_test, labels_test

