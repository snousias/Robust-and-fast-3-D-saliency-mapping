from definitions import *

# ========= Generic configuration =========#
print(80 * "=")
print('Initialize')
print(80 * "=")
rootdir = './'
modelsDir = 'data/'
modelFilename = 'joint.obj'
fullModelPath = rootdir + modelsDir + modelFilename
# fpath = fullModelPath.split(sep=".")[0]
fpath=rootdir
patchSide = 32
numOfElements = patchSide * patchSide
numberOfClasses = 4
saliencyDivisions = 64
useGuided = False
doRotate = True
doReadOBJ = True
rpcaneighbours = 60
pointcloudnn = 8
mode = "MESH"
patchSizeGuided = numOfElements

# ========= End of generic configuration =========#


# ========= Read models start=========#
print(80 * "=")
print('Read model data')
print(80 * "=")
(path, file) = os.path.split(fullModelPath)
filename, file_extension = os.path.splitext(file)
modelName = filename
mModelSrc = rootdir + modelsDir + modelName + '.obj'
print(modelName)
t = time.time()
presimplification = None

if mode == "MESH":
    saliencyGroundTrouthData = '_saliencyValues_of_centroids.csv'
    mModel = loadObj(mModelSrc)
    keyPredict = 'model_mesh' + modelName
    updateGeometryAttibutes(mModel, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided, computeDeltas=False,
                            computeAdjacency=False, computeVertexNormals=False)
    saliencyCombined = meshRPCASaliency(mModel,rpcaneighbours=rpcaneighbours)
    trm = time.time() - t
    print(80 * "=")
    print("Total time : " + str(trm))
    print(80 * "=")
    saliencyPerFace = saliencyCombined
    saliencyPerVertex = np.zeros((len(mModel.vertices)))
    for mVertexIndex, mVertex in enumerate(mModel.vertices):
        umbrella = [saliencyPerFace[mFaceIndex] for mFaceIndex in mVertex.neighbouringFaceIndices]
        umbrella = np.asarray(umbrella)
        saliencyPerVertex[mVertexIndex] = np.max(umbrella)
    np.savetxt(fpath +"/results/"+ modelName+saliencyGroundTrouthData, saliencyPerFace, delimiter=',', fmt='%10.3f')
    # --- color models ------
    for i, v in enumerate(mModel.vertices):
        # h=-((resultPerVertex[i] * 240*0.25))
        h = 0
        # s=1.0
        s = 0
        # v=1.0
        v = saliencyPerVertex[i]
        r, b, g = hsv2rgb(h, s, v)
        mModel.vertices[i] = mModel.vertices[i]._replace(color=np.asarray([r, g, b]))
    exportObj(mModel,fpath +"/results/"+ modelName+"_saliency" + ".obj", color=True)

if mode == "PC":
    saliencyGroundTrouthData = '_saliencyValues_of_points.csv'
    # mModel = loadObjPC(mModelSrc, nn=pointcloudnn, simplify=presimplification)
    # V, inds = computePointCloudNormals(mModel, pointcloudnn)
    Vertices = loadObjPointCloudFileToNumpyArray(mModelSrc, nn=pointcloudnn, simplify=presimplification)
    mModel, saliencyCombined = pointCloudToRPCASaliency(Vertices)
    saliencyPerVertex = saliencyCombined
    np.savetxt(fpath + saliencyGroundTrouthData, saliencyPerVertex, delimiter=',', fmt='%10.3f')
    # --- color models ------
    for i, v in enumerate(mModel.vertices):
        # h=-((resultPerVertex[i] * 240*0.25))
        h = 0
        # s=1.0
        s = 0
        # v=1.0
        v = saliencyPerVertex[i]
        r, b, g = hsv2rgb(h, s, v)
        mModel.vertices[i] = mModel.vertices[i]._replace(color=np.asarray([r, g, b]))
    exportObj(mModel, fpath +"/results/"+ modelName+"_saliency" + ".obj", color=True)
