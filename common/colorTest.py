import tensorflow as tf
import numpy as np
import random
import gzip
import tarfile
import pickle
import os
from six.moves import urllib
from commonReadOBJPointCloud import *
import scipy.io
from tensorflow.contrib.factorization import KMeans
import sklearn as sk
from scipy.spatial.distance import pdist, squareform
import hilbertcurve.hilbertcurve.hilbertcurve as hb
import pickle
import glob
import scipy.sparse as sp
from robust_pca import R_pca
from scipy.spatial.distance import directed_hausdorff
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt
import seaborn as sns
# ========= Generic configurations =========#
print(80 * "=")
print('Initialize')
print(80 * "=")
rootdir='F:/_Groundwork/FastSaliency/'
modelsDir='trainData/'
dataDir='data/'
fullModelPath="F:/_Groundwork/FastSaliency/trainData/armchair.obj"
patchSide=32
numOfElements = patchSide * patchSide
numberOfClasses=4
saliencyDivisions=64
useGuided = False
doRotate = True
doReadOBJ = True
rpcaneighbours=20
pointcloudnn=8
mode = "MESH"
saliencyGroundTrouthData = '_saliencyValues_of_centroids.csv'
patchSizeGuided = numOfElements

# ========= Read models =========#
print(80 * "=")
print('Read model data')
print(80 * "=")
(path, file) = os.path.split(fullModelPath)
filename, file_extension = os.path.splitext(file)
modelName=filename
mModelSrc = rootdir +modelsDir+ modelName + '.obj'
print(modelName)

if mode == "MESH":
    mModel = loadObj(mModelSrc)
    keyPredict = 'model_mesh' + modelName
    updateGeometryAttibutes(mModel, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided,computeDeltas=False,computeAdjacency=False,computeVertexNormals=False)


saliencyValuePath = rootdir + modelsDir + modelName + saliencyGroundTrouthData
saliencyPerFace = np.genfromtxt(saliencyValuePath, delimiter=',')


saliencyPerVertex=np.zeros((len(mModel.vertices)))
for mVertexIndex,mVertex in enumerate(mModel.vertices):
    umbrella=[saliencyPerFace[mFaceIndex] for mFaceIndex in mVertex.neighbouringFaceIndices]
    umbrella=np.asarray(umbrella)
    saliencyPerVertex[mVertexIndex]=np.max(umbrella)

#---- write to file ------
saliencyPerFace=saliencyPerFace/np.max(saliencyPerFace)
#np.savetxt(rootdir + modelsDir + modelName + saliencyGroundTrouthData, saliencyPerFace, delimiter=',',fmt='%10.3f')



# step = (1 / numberOfClasses)
# saliencyValueClass = (np.floor((saliencyPerVertex / step))).astype(int)
# saliencyValueClass = np.clip(saliencyValueClass, a_min = 0, a_max = 3)
# saliencyPerVertex = saliencyValueClass * 0.33 # For visualization purposes only


# --- color models ------
for i, v in enumerate(mModel.vertices):
    # h=-((resultPerVertex[i] * 240*0.25))
    v=saliencyPerVertex[i]
    if v<0:
        print('Error')
    r, b, g = hsv2rgb(0.0, 0.0, v)
    mModel.vertices[i] = mModel.vertices[i]._replace(color=np.asarray([r, g, b]))
exportObj(mModel, rootdir + modelsDir + modelName + "_gt_test" + ".obj", color=True)



