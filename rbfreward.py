import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
import os


# (some) hard-coded stuff
data_dir = '/home/warnellg/projects/social-navigation/'
data_file = 'simpleOut_1.txt'
d_large = 6  # feature dimension in data file
idx_phi = np.arange(2,6)  # per brian's email
r_bad = np.array((-997, -1001))  # per brian's email
r_fail = np.array((-999, -1000))  # per brian's email
r_newfail = -50.  # new reward for failed trial


# load data file
data_path = os.path.join(data_dir, data_file)
features = np.ndarray(shape=(d_large, 0))
rewards = []
with open(data_path) as data_file:
    philine = data_file.readline()
    rline = data_file.readline()
    while rline:
        # get features and reward
        phi = np.fromstring(philine, dtype='float', sep=' ').reshape(d_large, 1)
        r = np.fromstring(rline, dtype='float', sep=' ')

        # filter "bad" trials
        #    per brian's email, rewards of -997 and -1001 are bad trials
        if not np.isin(r, r_bad):
            features = np.hstack((features, phi))
            rewards = np.append(rewards, r)

        # read next lines
        philine = data_file.readline()
        rline = data_file.readline()


# do some data transformation
#     1. throw out irrelevant features
#     2. whiten features
#     3. rescale rewards - failed trials are less than -900 per brian's email
features = features[idx_phi, :]
feature_means = []
feature_vars = []
for i in range(np.shape(features)[0]):
    feature_means = np.append(feature_means, features[i, :].mean())
    feature_vars = np.append(feature_vars, features[i, :].var())
    features[i, :] = (features[i, :]-feature_means[i])/np.sqrt(feature_vars[i])
rewards[np.isin(rewards, r_fail)] = r_newfail


# reformat data for use with scikit-learn
X = features.transpose()
y = np.array(rewards)

# split data into training and testing
N = np.shape(X)[0]
idx = np.random.permutation(N)
N_train = int(np.floor(N*0.8))
N_test = int(N - N_train)
X_train = X[idx[0:N_train], :]
y_train = y[idx[0:N_train]]
X_test = X[idx[N_train:], :]
y_test = y[idx[N_train:]]


# fit kernel model
clf = KernelRidge(kernel='rbf', gamma=0.05)
clf.fit(X_train, y_train)


# report scaled mean absolute error for training and testing
err_train = clf.predict(X_train)-y_train
print('Mean absolute error (training): %f' % np.mean(np.abs(err_train)))
err_test = clf.predict(X_test)-y_test
print('Mean absolute error (test): %f' % np.mean(np.abs(err_test)))

print('\n')  # for debug breakpoint