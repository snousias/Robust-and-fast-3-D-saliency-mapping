
import os
from common.commonReadOBJPointCloud import *
from robust_pca import R_pca
import time


def pointCloudToRPCASaliency(Vertices, pointcloudnn, rpcaneighbours, presimplification):
    mModel = NumpyArrayToPointCloudStructure(Vertices, nn=pointcloudnn, simplify=presimplification)
    V, inds = computePointCloudNormals(mModel, pointcloudnn)
    Normals = []
    eigenvals = []
    print(80 * "=")
    print('Spectral saliency')
    print(80 * "=")
    iLen = len(mModel.vertices)
    for v_ind, f in enumerate(mModel.vertices):
        if v_ind % 2000 == 0:
            print('Extract patch information : ' + str(
                np.round((100 * v_ind / iLen), decimals=2)) + ' ' + '%')
        NormalsLine = np.empty(shape=[0, 3])
        patchFaces, rings = neighboursByVertex(mModel, v_ind, rpcaneighbours)
        for j in patchFaces:
            NormalsLine = np.append(NormalsLine, [mModel.vertices[j].normal], axis=0)
        nn = np.asarray(NormalsLine)
        conv1 = np.matmul(np.transpose(nn), nn)
        w, v = LA.eig(conv1)
        val = 1 / np.linalg.norm(w)
        eigenvals.append(val)
        NormalsLine = NormalsLine.ravel()
        Normals.append(NormalsLine)
    print(80 * "=")
    print('Geometric saliency')
    print(80 * "=")
    Normals = np.asarray(Normals)
    lmbda = 1 / np.sqrt(np.max(Normals.shape))
    mu = 10 * lmbda
    rpca = R_pca(Normals, lmbda=lmbda, mu=mu)
    LD, SD = rpca.fit(max_iter=700, iter_print=100, tol=1E-2)
    print("RPCA Shape:" + str(np.shape(Normals)) + "," + "Matrix Rank:" + str(
        np.linalg.matrix_rank(Normals)))
    print("LD Shape:" + str(np.shape(LD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(LD)) + " Max Val : " + str(
        np.max(LD)))
    print("SD Shape:" + str(np.shape(SD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(SD)) + " Max Val : " + str(
        np.max(SD)))

    CurvatureComponent = np.asarray(eigenvals)
    RPCAComponent = np.sum(np.abs(SD) ** 2, axis=-1) ** (1. / 2)
    print(np.shape(RPCAComponent))
    print(80 * "=")
    print('Combine')
    print(80 * "=")
    S1 = (RPCAComponent - RPCAComponent.min()) / (RPCAComponent.max() - RPCAComponent.min())
    E1 = (CurvatureComponent - CurvatureComponent.min()) / (CurvatureComponent.max() - CurvatureComponent.min())
    saliencyCombined = (S1 + E1) / 2
    saliencyCombined = saliencyCombined / np.max(saliencyCombined)

    return mModel, saliencyCombined


def meshRPCASaliency(mModel,rpcaneighbours):
    Normals = []
    eigenvals = []
    print(80 * "=")
    print('Spectral saliency')
    print(80 * "=")
    iLen = len(mModel.faces)
    for f_ind, f in enumerate(mModel.faces):
        if f_ind % 2000 == 0:
            print('Extract patch information : ' + str(
                np.round((100 * f_ind / iLen), decimals=2)) + ' ' + '%')
        NormalsLine = np.empty(shape=[0, 3])
        patchFaces, rings = neighboursByFace(mModel, f_ind, rpcaneighbours)
        for j in patchFaces:
            NormalsLine = np.append(NormalsLine, [mModel.faces[j].faceNormal], axis=0)
        nn = np.asarray(NormalsLine)
        conv1 = np.matmul(np.transpose(nn), nn)
        w, v = LA.eig(conv1)
        val = 1 / np.linalg.norm(w)
        eigenvals.append(val)
        NormalsLine = NormalsLine.ravel()
        Normals.append(NormalsLine)
    print(80 * "=")
    print('Geometric saliency')
    print(80 * "=")
    Normals = np.asarray(Normals)
    lmbda = 1 / np.sqrt(np.max(Normals.shape))
    mu = 10 * lmbda
    rpca = R_pca(Normals, lmbda=lmbda, mu=mu)
    LD, SD = rpca.fit(max_iter=700, iter_print=100, tol=1E-2)
    print("RPCA Shape:" + str(np.shape(Normals)) + "," + "Matrix Rank:" + str(
        np.linalg.matrix_rank(Normals)))
    print("LD Shape:" + str(np.shape(LD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(LD)) + " Max Val : " + str(
        np.max(LD)))
    print("SD Shape:" + str(np.shape(SD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(SD)) + " Max Val : " + str(
        np.max(SD)))
    CurvatureComponent = np.asarray(eigenvals)
    RPCAComponent = np.sum(np.abs(SD) ** 2, axis=-1) ** (1. / 2)
    print(np.shape(RPCAComponent))
    print(80 * "=")
    print('Combine')
    print(80 * "=")
    S1 = (RPCAComponent - RPCAComponent.min()) / (RPCAComponent.max() - RPCAComponent.min())
    E1 = (CurvatureComponent - CurvatureComponent.min()) / (CurvatureComponent.max() - CurvatureComponent.min())
    saliencyCombined = (S1 + E1) / 2
    saliencyCombined = saliencyCombined / np.max(saliencyCombined)
    return saliencyCombined
