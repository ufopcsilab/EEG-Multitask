import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.models import Model

from utils import load_test_samples


def calc_distances(feature1, feature2):
    feature1 = feature1.T
    xmax = np.amax(feature1,axis=0)
    xmin = np.amin(feature1,axis=0)
    x = feature1
    feature1 = (x-xmin)/(xmax-xmin)
    feature1 = feature1.T

    feature2 = feature2.T
    xmax = np.amax(feature2, axis=0)
    xmin = np.amin(feature2, axis=0)
    x = feature2
    feature2 = (x - xmin) / (xmax - xmin)
    feature2 = feature2.T

    # All against all euclidean distance
    dist = euclidean_distances(feature1, feature2)
    return dist


def get_scores(feature1, feature2, label1, label2):
    dist = calc_distances(feature1, feature2)
    genuine_list = []
    impostor_list = []
    for row in range(len(label1)):
        for col in range(row+1, len(label2)):
            if (label1[row] == label2[col]):
                genuine_list.append(dist[row, col])
            else:
                impostor_list.append(dist[row, col])

    genuine_scores = np.array(genuine_list)
    impostor_scores = np.array(impostor_list)

    return genuine_scores, impostor_scores


def calc_decidability(genuine_scores, impostor_scores):
    mean_genuine = np.mean(genuine_scores)
    mean_impostor = np.mean(impostor_scores)
    std_genuine = np.std(genuine_scores)
    std_impostor = np.std(impostor_scores)

    d = abs(mean_impostor - mean_genuine) / np.sqrt(0.5 * (std_genuine ** 2 + std_impostor ** 2))
    return d


def calc_eer(genuine, impostor, path='./', plot_det=True):
    dmin = np.amin(genuine)
    dmax = np.amax(impostor)

    resolu = 5000
    FMR = np.zeros(resolu)
    FNMR = np.zeros(resolu)
    t = np.linspace(dmin, dmax, resolu)

    for t_val in range(resolu):
        fm = np.sum(impostor <= t[t_val])
        FMR[t_val] = fm / len(impostor)
    for t_val in range(resolu):
        fnm = np.sum(genuine > t[t_val])
        FNMR[t_val] = fnm / len(genuine)

    if (plot_det==True):
        plt.figure()
        plt.plot(FMR, FNMR, color='darkorange', label='DET curve')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Match Rate')
        plt.ylabel('False NonMatch Rate')
        plt.title('Detection Error Trade-Off')
        plt.legend(loc="lower right")
        plt.savefig(path + r'EER.png', format='png')
        plt.show()
        plt.clf()

    abs_impostorfs = np.abs(FMR - FNMR)
    min_index = np.argmin(abs_impostorfs)
    eer = (FMR[min_index] + FNMR[min_index])/2
    thresholds = t[min_index]

    return eer, thresholds


if __name__ == '__main__':

    file_name = 'train_T1R1_9ch'
    test_tasks = ['R07']
    path = r'G:\Meu Drive\IC-EEG\EEG-Resultados\tasks\train_T1R1_9ch'
    channels= ['Af3.', 'Afz.', 'Af4.', 'C1..', 'Cz..', 'C2..', 'O1..', 'Oz..', 'O2..']

    model = load_model(path + r'/' + file_name + '.h5')
    model.summary()

    for task in test_tasks:
        print('\nTesting with: ', task)
        x_test, labels = load_test_samples(classes=108, test_task=task, channels=channels)

        ver_model = Model(inputs=model.input,
                          outputs=model.get_layer('encoding_layer').output)

        features = ver_model.predict(x_test)

        genuine_scores, impostor_scores = get_scores(features, features, labels, labels)
        d = calc_decidability(genuine_scores, impostor_scores)
        eer, threshold = calc_eer(genuine_scores, impostor_scores)

        print('Decidability = ' + str(d))
        print('EER = ' + str(eer))
        print('thresholds = ' + str(threshold))

