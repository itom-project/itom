"""
Class definition of the MLJ050 Labjack from Thorlabs.
It depends on the ftd2xx package (3 files: ftd2xx.py, _ftd2xx.py, defines.py).
It also requires the proper FTDI driver from http://www.ftdichip.com/Drivers/D2XX.htm
"""

import ftd2xx
import time
import codecs


# auxiliary functions
def littleendian2int(hexbytes):
    """
    Converts the bytes (little endian format) to integer.
    """
    if isinstance(
        hexbytes, int
    ):  # if hexbytes is only 1 byte long it is automaticly converted to an integer
        return hexbytes
    else:  # otherwise it has to be converted in reverse order
        revbytes = hexbytes[len(hexbytes) - 1 : None : -1]
        revhex = codecs.encode(revbytes, "hex")
        val = int(revhex, 16)
        if (
            val & (1 << (len(hexbytes) * 8 - 1))
        ) != 0:  # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << (len(hexbytes) * 8))  # compute negative value
        return val


def int2littleendian(integ, bytenum):
    """
    Converts an integer to bytes (little endian format). Bytenum defines the number of bytes.
    """
    if integ < 0:
        h2 = hex((1 << (bytenum * 8)) + integ)[2:]
    else:
        h = hex(integ)[2:]  # hex number of int (without leading '0x')
        if len(h) < 2 * bytenum:  # if h is too short, it is filled with 0s
            h2 = ""
            for zz in range(0, 2 * bytenum - len(h)):
                h2 = h2 + "0"
            h2 = h2 + h
        else:
            h2 = h
    revh = ""
    for bb in range(0, bytenum + 1):  # reverse hex
        revh = revh + h2[len(h2) - 2 * (bb + 1) : len(h2) - 2 * (bb)]
    return bytes.fromhex(revh)


class MLJ050:
    def __init__(self):
        # open FTDI Device, and set the parameters as given in the manual for MLJ050
        self.dev = ftd2xx.open()
        self.dev.setTimeouts(1000, 0)
        self.dev.setBaudRate(115200)
        self.dev.setDataCharacteristics(
            ftd2xx.BITS_8, ftd2xx.STOP_BITS_1, ftd2xx.PARITY_NONE
        )
        time.sleep(0.05)
        self.dev.purge(ftd2xx.PURGE_RX | ftd2xx.PURGE_TX)
        time.sleep(0.05)
        self.dev.resetDevice()
        self.dev.setFlowControl(ftd2xx.FLOW_RTS_CTS, 0, 0)
        self.dev.setRts()

        # ##########
        # Communicating with the device and setting the initialization parameters

        # This message is sent on start up to notify the controller of the source and destination addresses
        self.dev.write(bytes.fromhex("18 00 00 00 50 01"))

        # Request hardware info
        self.dev.write(bytes.fromhex("05 00 00 00 50 01"))
        self.HWinfo = self.dev.read(90)  # 90 bytes long
        self.SN = littleendian2int(
            self.HWinfo[6:10]
        )  # Serial number is given in bytes 7 to 10

        # Sent to enable or disable the specified drive channel. Channel 1, enable.
        self.dev.write(bytes.fromhex("10 02 01 01 50 01"))

        # The CONTROL IO connector on the rear panel of the unit exposes a number of digital outputs. This message is used to configure these digital outputs.
        self.dev.write(bytes.fromhex("13 02 00 ff 50 01"))

        # Set Trigger Mode -> Channel 1, Relative Move on Trigger
        self.dev.write(bytes.fromhex("00 05 01 10 50 01"))

        # Set the trapezoidal velocity parameters for the specified motor channel.
        # For stepper motor controllers the velocity is set in microsteps/sec and acceleration is set in microsteps/sec/sec
        # header (13 04 0e 00 d0 01)
        # channel ident (01 00)
        # MinVelocity (00 00 00 00) = 0
        # Acceleration (36 a6 01 00) = 108086 -> *90.9 wegen komischem Faktor = 9825017.4 -> ca. 7.99 mm/sec/sec (durch 1228800 microsteps)
        # Max Velocity (e0 52 cb 0b) = 197874400 -> /53.68 wegen komischem Faktor = 3686184.50 -> ca. 3mm/sec (durch 1228800 microsteps)
        # http://www.thorlabs.de/NewGroupPage9_PF.cfm?Guide=10&Category_ID=23&ObjectGroup_ID=4018
        # 1228800 microsteps = 1mm travel
        self.dev.write(
            bytes.fromhex("13 04 0e 00 d0 01 01 00 00 00 00 00 36 a6 01 00 e0 52 cb 0b")
        )

        # Set the velocity jog parameters for the specified motor channel
        # header(16 04 16 00 d0 01)
        # Channel (01 00)
        # Jog Mode (02 00) = 2 = single step jogging
        # Jog Step Size (00 60 09 00) = 0x00096000 = 614400 microsteps = 0.5 mm
        # Jog Min Velocity (f7 14 00 00) = 0x000014f7 = 5367 = ca.0 (?)
        # Jog Acceleration (c6 34 00 00) = 0.999 mm/sec/sec
        # Jog MAx Velocity (f4 70 ee 03) = 0.999 mm/sec
        # Stop Mode (02 00) = 2 = profiled stop deceleration (1= abrupt)
        self.dev.write(
            bytes.fromhex(
                "16 04 16 00 d0 01 01 00 02 00 00 60 09 00 f7 14 00 00 c6 34 00 00 f4 70 ee 03 02 00"
            )
        )

        # Set the limit switch parameters for the specified motor channel.
        # header (23 04 10 00 d0 01)
        # channel (01 00)
        # Clockwise Hardlimit (03 00) = 3 = Switch breaks on contact
        # Counter Clockwise Hardlimit (03 00) = s.o.
        # CW softlimit (00 40 38 00)
        # CCW softlimit (00 c0 12 00)
        # Limit Mode (01 00) = Ignore Limit (?!!?)
        self.dev.write(
            bytes.fromhex(
                "23 04 10 00 d0 01 01 00 03 00 03 00 00 40 38 00 00 c0 12 00 01 00"
            )
        )

        # Set Power Parameters.
        self.dev.write(bytes.fromhex("26 04 06 00 d0 01 01 00 0f 00 3c 00"))

        # Set the general move parameters for the specified motor channel
        # At this time this refers specifically to the backlash settings.
        self.dev.write(bytes.fromhex("3a 04 06 00 d0 01 01 00 00 68 01 00"))

        # Set the home parameters for the specified motor channel.
        # header + channel (8byte)
        # home dir (02 00) = ignored, always positive
        # limit switch (01 00) = ignored
        # Home Velocity (f4 70 ee 03) = 0.999 mm/sec
        # Offset Distance (00 c0 03 00) = not used
        self.dev.write(
            bytes.fromhex("40 04 0e 00 d0 01 01 00 02 00 01 00 f4 70 ee 03 00 c0 03 00")
        )

        # Set the relative move parameters for the specified motor channel.
        # The only significant parameter at this time is the relative move distance itself.
        # This gets stored by the controller and is used the next time a relative move is initiated
        self.dev.write(bytes.fromhex("45 04 06 00 d0 01 01 00 00 0a 00 00"))

        # Set the absolute move parameters for the specified motor channel.
        # The only significant parameter at this time is the absolute move position itself.
        # This gets stored by the controller and is used the next time an absolute move is initiated.
        self.dev.write(bytes.fromhex("50 04 06 00 d0 01 01 00 00 00 00 00"))

        # Set bow index.
        self.dev.write(bytes.fromhex("f4 04 04 00 d0 01 01 00 00 00"))

        # Set joystick parameters
        self.dev.write(
            bytes.fromhex(
                "e6 04 14 00 d0 01 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 02 00"
            )
        )

        # Similar Settings for Channel 2 (Joystick???)
        # enable channel 2:
        self.dev.write(bytes.fromhex("10 02 02 01 50 01"))
        # configure digital output
        self.dev.write(bytes.fromhex("13 02 00 ff 50 01"))
        # set trigger mode
        self.dev.write(bytes.fromhex("00 05 02 10 50 01"))
        # set trapezoidal velocity parameters
        self.dev.write(
            bytes.fromhex("13 04 0e 00 d0 01 02 00 00 00 00 00 83 7c 00 00 00 f4 01 00")
        )
        # set velocity jog parameters
        self.dev.write(
            bytes.fromhex(
                "16 04 16 00 d0 01 02 00 02 00 00 7d 00 00 00 00 00 00 83 7c 00 00 00 7d 00 00 00 00"
            )
        )
        # set limit switch parameters
        self.dev.write(
            bytes.fromhex(
                "23 04 10 00 d0 01 02 00 03 00 01 00 9d ff ff ff 9d ff ff ff 01 00"
            )
        )
        # set power parameters
        self.dev.write(bytes.fromhex("26 04 06 00 d0 01 02 00 14 00 64 00"))
        # set general move parameters
        self.dev.write(bytes.fromhex("3a 04 06 00 d0 01 02 00 00 0a 00 00"))
        # set home parameters
        self.dev.write(
            bytes.fromhex("40 04 0e 00 d0 01 02 00 02 00 01 00 00 7d 00 00 00 32 00 00")
        )
        # set relative move parameters
        self.dev.write(bytes.fromhex("45 04 06 00 d0 01 02 00 00 0a 00 00"))
        # set absolute move parameter
        self.dev.write(bytes.fromhex("50 04 06 00 d0 01 02 00 00 00 00 00"))
        # set bow index
        self.dev.write(bytes.fromhex("f4 04 04 00 d0 01 02 00 00 00"))
        # set joystick parameters
        self.dev.write(
            bytes.fromhex(
                "e6 04 14 00 d0 01 02 00 4a 0c 02 00 e1 7a 14 00 13 01 00 00 13 01 00 00 01 00"
            )
        )

    def __del__(self):
        # Closing the FTDI device, deleting its parameter.
        self.dev.close()
        del self.dev

    def write(self, bb):
        # send byte to the device
        return self.dev.write(bb)

    def read(self, bnum):
        # reads number of bytes (bnum) from the device
        return self.dev.read(bnum)

    def jogUp(self):
        # move the stage a defined distance (jog step) upwards
        self.dev.write(bytes.fromhex("6a 04 01 01 50 01"))
        while len(self.dev.read(22)) == 0:
            print("moving...")

    def jogDown(self):
        # move the stage a defined distance (jog step) upwards
        self.dev.write(bytes.fromhex("6a 04 01 02 50 01"))
        while len(self.dev.read(22)) == 0:
            print("moving...")

    def moveHome(self):
        self.dev.write(bytes.fromhex("43 04 01 00 50 01"))
        while len(self.dev.read(6)) == 0:
            print("moving home...")

    def moveRel(self, distance):
        # moves the stage relative to the actual position by the distance in mm
        steps = 1228800 * distance  # 1228800 microsteps = 1mm travel
        header = bytes.fromhex("48 04 06 00 d0 01")
        chan = int2littleendian(1, 2)
        reldis = int2littleendian(int(steps), 4)
        sendhex = header + chan + reldis
        self.dev.write(sendhex)
        while len(self.dev.read(20)) == 0:
            print("moving ...")

    def moveAbs(self, position):
        # moves the stage to the absolut position in mm
        steps = 1228800 * position  # 1228800 microsteps = 1mm travel
        header = bytes.fromhex("53 04 06 00 d0 01")
        chan = int2littleendian(1, 2)
        poshex = int2littleendian(int(steps), 4)
        sendhex = header + chan + poshex
        self.dev.write(sendhex)
        while len(self.dev.read(20)) == 0:
            print("moving ...")

    def getPos(self):
        # get the position of the stage in mm
        self.dev.write(bytes.fromhex("80 04 01 00 d0 01"))  # request status update
        status = self.dev.read(20)
        poshex = status[8:12]  # position from byte 8-11
        posstep = littleendian2int(poshex)
        return posstep / 1228800
