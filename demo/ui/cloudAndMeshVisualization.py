import numpy as np

try:
    from itom import pointCloud
except Exception as ex:
    ui.msgInformation("PointCloud missing", "your itom version is compiled without support of pointClouds")
    raise ex

#create a data objects with X, Y and Z values of a topography
# as well as a 2.5D topography in terms of a data object
[X,Y] = np.meshgrid(np.arange(0,50,0.1),np.arange(0,50,0.1))
Z = np.sin(X*2)+np.cos(Y*0.5)
I = np.random.rand(*X.shape) #further intensity
C = dataObject.randN([X.shape[0],X.shape[1]], 'rgba32') #further color information
topography = dataObject(Z).astype('float32')
topography.axisScales = (0.1,0.1)
topography[0,0]=float('nan')

mesh_quads = polygonMesh.fromTopography(topography)
mesh_triangles = polygonMesh.fromTopography(topography, triangulationType = 1)

[i,h] = plot(mesh_quads, "vtk3dvisualizer")
h.call("addMesh", mesh_triangles, "mesh_triangles")