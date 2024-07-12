macro(itom_plugin_option PLUGIN_ID)

    set(PLUGINS_LIST	# Legend: X = OFF, D = Default, S = Setup, T = Test
    "+-------------------------------+-----------------------------------+"
    "| **Plugin**                    |  Win  | macOS | Ubu2404 | Rasbian |"
    "+===============================+===================================+"
	"| PLUGIN_LIBUSB                 |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_aerotechA3200          |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_aerotechEnsemble       |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_andorSDK3              |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_AvantesAvaSpec         |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_AVTVimba               |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_BasicFilters           |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_BasicGPLFilters        |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_HidApi                 |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_cmu1394                |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_CommonVisionBlox       |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_CyUSB                  |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_dataobjectarithmetic   |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_DataObjectIO           |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_DIC                    |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_dispWindow             |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_DslrRemote             |   T   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_DslrRemote2            |   T   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_DummyGrabber           |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_DummyMotor             |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_DemoAlgorithms         |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_FFTWFilters            |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_FileGrabber            |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_FittingFilters         |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_FireGrabber            |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_FirgelliLAC            |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_FringeProj             |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_GenICam                |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_GLDisplay              |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_GWInstekPSP            |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_HBMSpider8             |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_Holography             |   T   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_IDSuEye                |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_LeicaMotorFocus        |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_LibModbus              |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_MeasurementComputing   |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_MSMediaFoundation      |   D   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_NerianSceneScanPro     |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_NanotecStepMotor       |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_NEWPORT_2936           |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_NEWPORT_CONEXLDS       |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_NEWPORT_SMC100         |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_niDAQmx                |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_NITWidySWIR            |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_OceanOpticsSpec        |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_OpenCVFilters          |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_OpenCVFilters_Nonfree  |   T   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_OpenCVGrabber          |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_OphirPowermeter        |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_PclTools               |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_PCOCamera              |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_PCOSensicam            |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_PCOPixelFly            |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_PiezosystemJena_NV40_1 |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_PGRFlyCapture          |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_PIHexapodCtrl          |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_PIPiezoCtrl            |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_PI_GCS2                |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_PMD_PICO               |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_QCam                   |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_QuantumComposer        |   D   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_rawImport              |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_Roughness              |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_SerialIO               |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_SuperlumBS             |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_SuperlumBL             |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_ST8SMC4USB             |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_ThorlabsBP             |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_ThorlabsBDCServo       |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_ThorlabsFF             |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_ThorlabsTCubeTEC       |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_ThorlabsKCubeDCServo   |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_ThorlabsKCubeIM        |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_ThorlabsKCubePA        |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_ThorlabsCCS            |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_ThorlabsISM            |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_ThorlabsDCxCam         |   T   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_ThorlabsPowerMeter     |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_UhlRegister            |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_UhlText                |   D   |   D   |    D    |    D    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_USBMotion3XIII         |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_V4L2                   |   X   |   X   |    T    |    T    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_Vistek                 |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_VRMagic                |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_x3pio                  |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_Xeneth                 |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
	"| PLUGIN_XIMEA                  |   S   |   X   |    X    |    X    |"
	"+-------------------------------+-----------------------------------+"
    "| **Plugin**                    |  Win  | macOS | Ubu2404 | Rasbian |"
)

    set(PATTERN "${PLUGIN_ID}.*$")

    # get column index
	if(WIN32)
		set(INDEX 1)
		message(STATUS "Operating System: Windows")
	elseif(APPLE)
		set(INDEX 2)
		message(STATUS "Operating System: macOS")
	elseif(UNIX)
		file(READ "/etc/os-release" OS_RELEASE_CONTENTS)
		if(OS_RELEASE_CONTENTS MATCHES "Ubuntu")
			set(INDEX 3)
			message(STATUS "Operating System: Ubuntu")
		elseif(OS_RELEASE_CONTENTS MATCHES "Raspbian")
			set(INDEX 4)
			message(STATUS "Operating System: Raspbian")
		endif()
	else()
		message(FATAL_ERROR "Operating System not support. Itom is available for Win, macOS, Ubuntu2404 and Rasbian only.")    
	endif()

    foreach(PLUGIN_ROW ${PLUGINS_LIST})
        # get row
        string(REGEX MATCH ${PATTERN} MATCHSTRING ${PLUGIN_ROW})
        if(MATCHSTRING)

            string(REPLACE "|" ";" SPLIT_LIST "${MATCHSTRING}")
            list(GET SPLIT_LIST ${INDEX} ELEMENT)
            string(STRIP "${ELEMENT}" VALUE)

            # case DEFAULT
            if(VALUE STREQUAL "D")
                set(BUILD_OPTION ON)
            # case SETUP
            elseif(ITOM_BUILD_SETUP AND (VALUE STREQUAL "D" OR VALUE STREQUAL "S"))
                set(BUILD_OPTION ON)
            # case TEST
            elseif(ITOM_BUILD_TEST AND (VALUE STREQUAL "D" OR VALUE STREQUAL "S" OR VALUE STREQUAL "T"))
                set(BUILD_OPTION ON)
            else()
                set(BUILD_OPTION OFF)
            endif()
        endif(MATCHSTRING)
    endforeach()


    option(${PLUGIN_ID} "Build with this plugin." ${BUILD_OPTION})

endmacro()
