import unittest
import numpy as np
import pickle
import io

try:
    from itom import pointCloud, point, polygonMesh, dataObject
    hasPCL = True
except (ModuleNotFoundError, ImportError):
    hasPCL = False

if hasPCL:
    class PointCloudPickle(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            pass

        def test_picklePointCloud(self):
            """dumps and reloads a point cloud and compare both.

            Pickling is also used by loadIDC, saveIDC."""
            cloud = pointCloud(point(point.PointXYZ, (0.0, -1.0, 2.0)))
            bytes_io = io.BytesIO()
            # should pass
            pickle.dump(cloud, bytes_io)

            # move ptr in buffer to start again
            bytes_io.seek(0)

            cloud2 = pickle.load(bytes_io)

            self.assertEqual(cloud.size, cloud2.size)

        def test_picklePolygonMesh(self):
            """dumps and reloads a polygon mesh and compare both.

            Pickling is also used by loadIDC, saveIDC."""
            mesh = polygonMesh.fromTopography(dataObject.randN([10,10], 'float32'))
            bytes_io = io.BytesIO()
            # should pass
            pickle.dump(mesh, bytes_io)

            # move ptr in buffer to start again
            bytes_io.seek(0)

            mesh2 = pickle.load(bytes_io)

            self.assertTrue(np.all(mesh.getPolygons()==mesh2.getPolygons()))

else:
    class PointCloudPickle(unittest.TestCase):
        pass


if __name__ == '__main__':
    unittest.main(module='pointcloud_pickle', exit=False)
