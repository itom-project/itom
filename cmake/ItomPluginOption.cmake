macro(itom_plugin_option PLUGIN_ID)

    set(PLUGINS_LIST	# Legend: X = OFF, D = Default, S = Setup, T = Test
    "+-------------------------------+-------------------------------------------+"
    "| **Plugin**                    | Win11 | Win10 | macOS | Ubu2404 | Rasbian |"
    "+===============================+===========================================+"
	"| PLUGIN_LIBUSB                 |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_aerotechA3200          |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_aerotechEnsemble       |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_andorSDK3              |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_AvantesAvaSpec         |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_AVTVimba               |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_BasicFilters           |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_BasicGPLFilters        |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_HidApi                 |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_cmu1394                |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_CommonVisionBlox       |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_CyUSB                  |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_dataobjectarithmetic   |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_DataObjectIO           |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_DIC                    |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_dispWindow             |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_DslrRemote             |   X   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_DslrRemote2            |   X   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_DummyGrabber           |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_DummyMotor             |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_DemoAlgorithms         |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_FFTWFilters            |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_FileGrabber            |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_FittingFilters         |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_FireGrabber            |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_FirgelliLAC            |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_FringeProj             |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_GenICam                |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_GLDisplay              |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_GWInstekPSP            |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_HBMSpider8             |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_Holography             |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_IDSuEye                |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_LeicaMotorFocus        |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_LibModbus              |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_MeasurementComputing   |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_MSMediaFoundation      |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_NerianSceneScanPro     |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_NanotecStepMotor       |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_NEWPORT_2936           |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_NEWPORT_CONEXLDS       |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_NEWPORT_SMC100         |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_niDAQmx                |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_NITWidySWIR            |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_OceanOpticsSpec        |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_OpenCVFilters          |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_OpenCVFilters_Nonfree  |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_OpenCVGrabber          |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_OphirPowermeter        |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_PclTools               |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_PCOCamera              |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_PCOSensicam            |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_PCOPixelFly            |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_PiezosystemJena_NV40_1 |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_PGRFlyCapture          |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_PIHexapodCtrl          |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_PIPiezoCtrl            |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_PI_GCS2                |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_PMD_PICO               |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_QCam                   |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_QuantumComposer        |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_rawImport              |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_Roughness              |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_SerialIO               |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_SuperlumBS             |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_SuperlumBL             |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_ST8SMC4USB             |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_ThorlabsBP             |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_ThorlabsBDCServo       |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_ThorlabsFF             |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_ThorlabsTCubeTEC       |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_ThorlabsKCubeDCServo   |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_ThorlabsKCubeIM        |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_ThorlabsKCubePA        |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_ThorlabsCCS            |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_ThorlabsISM            |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_ThorlabsDCxCam         |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_ThorlabsPowerMeter     |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_UhlRegister            |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_UhlText                |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_USBMotion3XIII         |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_V4L2                   |   D   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_Vistek                 |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_VRMagic                |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_x3pio                  |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_Xeneth                 |   S   |   0   |   0   |    0    |    0    |"
	"+-------------------------------+-------------------------------------------+"
	"| PLUGIN_XIMEA                  |   S   |   0   |   0   |    0    |    0    |"
)

    set(PATTERN "${PLUGIN_ID}.*$")

    # get column index
    if(WIN32)
        set(INDEX 1)
    endif(WIN32)

    if(APPLE)
        set(INDEX 3)
    endif(APPLE)

    if(UNIX)
        set(INDEX 4)
    endif(UNIX)

    foreach(PLUGIN_ROW ${PLUGINS_LIST})
        # get row
        string(REGEX MATCH ${PATTERN} MATCHSTRING ${PLUGIN_ROW})
        if(MATCHSTRING)
            message(STATUS "MATCHSTRING: ${MATCHSTRING}")

            string(REPLACE "|" ";" SPLIT_LIST "${MATCHSTRING}")
            message(STATUS "SPLIT_LIST: ${SPLIT_LIST}")

            list(GET SPLIT_LIST ${INDEX} ELEMENT)
            message(STATUS "ELEMENT: ${ELEMENT}")

            string(STRIP "${ELEMENT}" VALUE)
            message(STATUS "VALUE: ${VALUE}")

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

			message(STATUS "${PLUGIN_ID} BUILD_OPTION: ${BUILD_OPTION}")

        endif(MATCHSTRING)
    endforeach()


    option(${PLUGIN_ID} "Build with this plugin." ${BUILD_OPTION})

endmacro()
