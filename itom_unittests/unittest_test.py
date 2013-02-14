import unittest

class TestDataObject(unittest.TestCase):
	
	def setUp(self):
		a=1
	
	def test_1(self):
		a=dataObject([4,5],dtype='uint8')
		a.ones([4,5])
		b=a.copy()
		self.assertEqual(a,b)
		
	def test_2(self):
		pass

#class TestSequenceFunctions(unittest.TestCase):
#
#    def setUp(self):
#        self.seq = range(10)
#        a=dataObject()
#
#    def test_shuffle2(self):
#        # make sure the shuffled sequence does not lose any elements
#        #random.shuffle(self.seq)
#        #self.seq.sort()
#        self.assertEqual(self.seq, range(10))
#
#        # should raise an exception for an immutable sequence
#        self.assertRaises(TypeError, random.shuffle, (1,2,3))
#        self.assertTrue(1)#
#
#    def test_choice(self):
#        element = random.choice(self.seq)
#        self.assertTrue(element in self.seq)
#
#    def test_sample(self):
#        with self.assertRaises(ValueError):
#            random.sample(self.seq, 20)
#        for element in random.sample(self.seq, 5):
#            self.assertTrue(element in self.seq)

suite = unittest.TestLoader().loadTestsFromTestCase(TestDataObject)
unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)