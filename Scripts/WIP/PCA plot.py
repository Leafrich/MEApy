import os
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from Functions import *
import McsPy.McsData

def extractMethods(item):
    # Methods extraction from class
    _ = [method for method in dir(item) if callable(getattr(item, method))]
    for i in range(len(_)):
        print(_[i])

"""
Loading data into panda
"""

electrode_id = 4
timeStart = 0
timeStop = 300

rootDir = os.path.dirname(os.path.abspath(os.path.join(__file__ ,"../..")))
data = f'{rootDir}\\Data\\spk_PCA\\20230605_Cortex_pMEA_001_mwc.h5'
file = McsPy.McsData.RawData(data)
segments = file.recordings[0].segment_streams[0].segment_entity
# ids = [c.channel_id for c in segments.segment_infos.values()]
# channel_id = ids[electrode_id]
ids = list(segments.keys())
print(ids[0])
print(segments[ids[0]].data[1])


# scaler = StandardScaler()
# scaled_cutouts = scaler.fit_transform(cutouts)
#
# pca = PCA()
# pca.fit(scaled_cutouts)
#
# # j = 0
# # n = 0
# # for i in range(len(pca.explained_variance_ratio_)):
# #     j += pca.explained_variance_ratio_[i]
# #     n += 1
# #     if j >= .2:
# #         break
#
# transformed_3d = pca.fit_transform(scaled_cutouts)
# print("Dim transformed %s" % str(np.shape(transformed_3d)))
#
# # print("%s Compenents 0.6 variabilty explained" % n)
# n_components = 3
#
# gmm = GaussianMixture(n_components=n_components, n_init=10)
# labels = gmm.fit_predict(transformed_3d)
#
# _ = plt.figure(figsize=(8, 8))
# for i in range(n_components):
#     idx = labels == i
#     _ = plt.plot(transformed_3d[idx, 0], transformed_3d[idx, 1], '.')
#     _ = plt.title('Cluster assignments by a GMM')
#     _ = plt.xlabel('Principal Component 1')
#     _ = plt.ylabel('Principal Component 2')
#     _ = plt.axis('tight')
# plt.show()
#
# _ = plt.figure(figsize=(12, 6))
# for i in range(n_components):
#     idx = labels == i
#     color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
#     plot_waveforms(cutouts[idx, :], fs, pre, post, n=100, color=color, show=False)
# plt.show()