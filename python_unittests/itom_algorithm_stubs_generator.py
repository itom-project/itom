import itomAlgorithmsStubsGen as algoStubsGen
import unittest
from typing import Tuple, Dict, List, Optional
import warnings
import itom



class ItomAlgorithmsStubsGenTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_generate_hash(self):
        checksum = algoStubsGen.generateAlgorithmHash()
        self.assertFalse(checksum == b"")

    def test_parse_algorithm_def(self):
        algos = itom.filterHelp("", dictionary=1, furtherInfos=1)
        algoItems = []
        found = False
        
        for algo in algos:
            if algo == "centroid1D":
                found = True
                docstring = algoStubsGen.parseAlgorithmString(algos[algo])
                self.assertTrue(docstring.startswith("def centroid1D(sourceImage: dataObject, destCOG: dataObject, destIntensity: dataObject, pvThreshold: float = 0.0, dynamicThreshold: float = 0.5, lowerThreshold: float = -1.7976931348623157e+308, columnWise: int = 0) -> None:"))
                break

        self.assertTrue(found)

    def test_read_pcl_version(self):
        pclVersion = itom.version(dictionary=True)["itom"]["PCL_Version"]
        self.assertTrue(pclVersion != "")

if __name__ == '__main__':
    unittest.main(module='itom_algorithm_stubs_generator', exit=False)