import unittest
from itom import dataIO, dataObject

def channelSpecificFunc1():
    cam = dataIO("DummyMultiChannelGrabber")
    cam.setParam('defaultChannel','channel_1')
    cam.setParam('channelSpecificParameter',200)

def test_invalidDefaultChannelErrorFunc():
    cam = dataIO("DummyMultiChannelGrabber")
    cam.setParam('defaultChannel', 'xyz')

class MultiChannelDummyGrabberTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def channelSpecificFunc1():
        cam = dataIO("DummyMultiChannelGrabber")
        cam.setParam('defaultChannel','Channel_1')
        cam.setParam('channelSpecificParameter',200)

    def test_setAndGetParamViaDefaultChannel(self):
        cam = dataIO("DummyMultiChannelGrabber")
        cam.setParam('defaultChannel', 'channel_0')
        originalRoi = cam.getParam('roi')
        cam.setParam('defaultChannel', 'channel_1')
        cam.setParam('roi', [0,0,40,40])
        cam.setParam('defaultChannel', 'channel_0')
        channel0Roi = cam.getParam('roi')
        cam.setParam('defaultChannel', 'channel_1')
        channel1Roi = cam.getParam('roi')

        self.assertEqual(originalRoi, channel0Roi)
        self.assertNotEqual(originalRoi, channel1Roi)

    def test_invalidDefaultChannelErrorTest(self):
        self.assertRaises(RuntimeError, test_invalidDefaultChannelErrorFunc)

    def test_invalidDefaultChannelTest(self):
        cam = dataIO("DummyMultiChannelGrabber")
        paramList = cam.getParamList()
        originalParamDict = {}
        for elem in paramList:
            originalParamDict[elem] = cam.getParam(elem)
        try:
            cam.setParam('defaultChannel', 'xyz')
        except RuntimeError:
            pass
        testParamDict = {}
        for elem in paramList:
            testParamDict[elem] = cam.getParam(elem)
        self.assertEqual(originalParamDict, testParamDict)


    def test_getValByDict(self):
        cam=dataIO("DummyMultiChannelGrabber")
        cam.startDevice()
        cam.acquire()
        channelList=cam.getParam('channelList')
        roi={}
        getValDict={}
        for elem in channelList:
            cam.setParam("defaultChannel", elem)
            roi[elem] = cam.getParam('roi')
            getValDict[elem]=dataObject()
        cam.getVal(getValDict)
        for key in roi:
            self.assertEqual(roi[key][-2], getValDict[key].shape[-1])
            self.assertEqual(roi[key][-1], getValDict[key].shape[-2])

    def test_copyValByDict(self):
        cam=dataIO("DummyMultiChannelGrabber")
        cam.startDevice()
        cam.acquire()
        channelList=cam.getParam('channelList')
        roi={}
        getValDict={}
        for elem in channelList:
            cam.setParam('defaultChannel',elem)
            roi[elem] = cam.getParam('roi')
            getValDict[elem]=dataObject()
        cam.copyVal(getValDict)
        for key in roi:
            self.assertEqual(roi[key][-2], getValDict[key].shape[-1])
            self.assertEqual(roi[key][-1], getValDict[key].shape[-2])


    def test_channelSpecific(self):
        self.assertRaises(RuntimeError, channelSpecificFunc1)



    if __name__ == '__main__':
        unittest.main(module='multichannel_test', exit=False)
