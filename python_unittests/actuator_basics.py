import unittest
import itom
import numpy as np

# these tests require the DummyMotor plugin


class ActuatorBasicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_setOrigin(self):
        motor1 = itom.actuator("DummyMotor", numAxis=1)
        motor3 = itom.actuator("DummyMotor", numAxis=3)

        # check integer arguments
        self.assertIsNone(motor1.setOrigin(0))
        self.assertIsNone(motor1.setOrigin(np.int32(0)))

        self.assertIsNone(motor3.setOrigin(2))
        self.assertIsNone(motor3.setOrigin(np.int32(2)))

        self.assertIsNone(motor3.setOrigin(0, 2))
        self.assertIsNone(motor3.setOrigin(np.int32(0), 1, np.int(2)))

        # check wrong argument types
        with self.assertRaises(TypeError):
            motor1.setOrigin(0.1)

        with self.assertRaises(TypeError):
            motor1.setOrigin(np.float32(0.0))

        with self.assertRaises(TypeError):
            motor1.setOrigin(np.float64(0.0))

        with self.assertRaises(ValueError):
            motor1.setOrigin()

        with self.assertRaises(ValueError):
            motor3.setOrigin()

    def test_calib(self):
        """
        Reduce number of tests, since calib of DummyMotor is a 5sec operation always.
        """
        motor1 = itom.actuator("DummyMotor", numAxis=1)
        motor3 = itom.actuator("DummyMotor", numAxis=3)

        # check integer arguments
        self.assertIsNone(motor3.calib(np.int32(0), 1, np.int(2)))

        # check wrong argument types
        with self.assertRaises(TypeError):
            motor1.calib(0.1)

        with self.assertRaises(TypeError):
            motor1.calib(np.float32(0.0))

        with self.assertRaises(TypeError):
            motor1.calib(np.float64(0.0))

        with self.assertRaises(ValueError):
            motor1.calib()

        with self.assertRaises(ValueError):
            motor3.calib()

    def test_getPos(self):
        motor1 = itom.actuator("DummyMotor", numAxis=1)
        motor3 = itom.actuator("DummyMotor", numAxis=3)

        # check integer arguments
        self.assertEqual(type(motor1.getPos(0)), float)
        self.assertEqual(type(motor1.getPos(np.int32(0))), float)

        self.assertEqual(type(motor3.getPos(2)), float)
        self.assertEqual(type(motor3.getPos(np.int32(2))), float)

        self.assertEqual(len(motor3.getPos(0, 2)), 2)
        self.assertEqual(len(motor3.getPos(np.int32(0), 1, np.int(2))), 3)

        # check wrong argument types
        with self.assertRaises(TypeError):
            motor1.getPos(0.1)

        with self.assertRaises(TypeError):
            motor1.getPos(np.float32(0.0))

        with self.assertRaises(TypeError):
            motor1.getPos(np.float64(0.0))

        with self.assertRaises(ValueError):
            motor1.getPos()

        with self.assertRaises(ValueError):
            motor3.getPos()

    def test_setPos(self):
        motor1 = itom.actuator("DummyMotor", numAxis=1)
        motor3 = itom.actuator("DummyMotor", numAxis=3)

        self.assertIsNone(motor1.setPosAbs(np.int32(0), np.float(0.05)))
        self.assertEqual(motor1.getPos(0), 0.05)
        self.assertIsNone(
            motor3.setPosAbs(np.int32(0), 1, 1, np.int64(-1), 2, np.float64(0.001))
        )
        self.assertTupleEqual(motor3.getPos(0, 1, 2), (1, -1, 0.001))
        self.assertIsNone(
            motor3.setPosRel(np.int32(0), 1, 1, np.int64(-1), 2, np.float64(0.001))
        )
        self.assertTupleEqual(motor3.getPos(0, 1, 2), (2, -2, 0.002))


if __name__ == "__main__":
    unittest.main(module="actuator_basics", exit=False)
