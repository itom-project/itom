# ##############################################################################
# SETUP Configuration to define CMAKE Variables
# to be used for the PCL Detection in ITOM

if(NOT EXISTS "${EIGEN_ROOT}")
    if(EXISTS "$ENV{EIGEN_ROOT}")
        set(EIGEN_ROOT $ENV{EIGEN_ROOT} CACHE PATH "Path to the Eigen3 Directory")
    else()
        set(EIGEN_ROOT "EIGEN_ROOT-NOTFOUND" CACHE PATH "Path to the Eigen3 Directory" FORCE)
    endif()
endif()

# PCL >= 1.14 requires the Eigen3 CMake package in config mode (Eigen3Config.cmake),
# so derive Eigen3_DIR from EIGEN_ROOT if the user did not set it explicitly.
if(EXISTS "${EIGEN_ROOT}" AND NOT EXISTS "${Eigen3_DIR}")
    find_path(Eigen3_DIR
        NAMES Eigen3Config.cmake eigen3-config.cmake
        HINTS "${EIGEN_ROOT}" "${EIGEN_ROOT}/.."
        PATH_SUFFIXES
        share/eigen3/cmake
        share/cmake/eigen3
        lib/cmake/eigen3
        cmake
        DOC "The directory containing a CMake configuration file for Eigen3."
        NO_DEFAULT_PATH)

    if(NOT EXISTS "${Eigen3_DIR}")
        message(WARNING "No Eigen3Config.cmake found below EIGEN_ROOT='${EIGEN_ROOT}'. \
PCL >= 1.14 requires an installed Eigen3 CMake package. Configure and install the Eigen \
sources (cmake -S <eigen-src> -B <build> -DCMAKE_INSTALL_PREFIX=${EIGEN_ROOT} && cmake --install <build>) \
or set Eigen3_DIR to the folder that contains Eigen3Config.cmake.")
    endif()
endif()

# Boost use only Static Libs, also used as a search parameter
set(Boost_USE_STATIC_LIBS ON)

if(NOT EXISTS "${Boost_INCLUDE_DIR}")
    if(EXISTS "$ENV{BOOST_ROOT}")
        set(Boost_INCLUDE_DIR $ENV{BOOST_ROOT} CACHE PATH "Path to the BOOST Directory")
    else()
        set(Boost_INCLUDE_DIR "Boost_INCLUDE_DIR-NOTFOUND" CACHE PATH "Path to the BOOST Directory" FORCE)
    endif()
endif()

if(NOT EXISTS "${PCL_DIR}")
    if(EXISTS "$ENV{PCL_ROOT}")
        set(PCL_DIR $ENV{PCL_ROOT} CACHE PATH "Path to the Point Cloud Directory")
    else()
        set(PCL_DIR "PCL_DIR-NOTFOUND" CACHE PATH "Path to the Point Cloud Directory" FORCE)
    endif()
endif()

if(WIN32)
    get_filename_component(PCL_CMAKE_FOLDER "${PCL_DIR}" NAME)

    if("${PCL_CMAKE_FOLDER}" STREQUAL "cmake")
        set(PCL_CMAKE_DIR "${PCL_DIR}")
    else()
        set(PCL_CMAKE_DIR "${PCL_DIR}/cmake")
    endif()

    if(NOT EXISTS "${PCL_DIR}" OR NOT EXISTS "${Boost_INCLUDE_DIR}" OR NOT EXISTS "${EIGEN_ROOT}")
        message(FATAL_ERROR "Dependencies Missing for Point-Cloud Library. Please make sure that PCL_DIR, Boost_INCLUDE_DIR and EIGEN_ROOT are well defined (PCL_DIR='${PCL_DIR}', Boost_INCLUDE_DIR='${Boost_INCLUDE_DIR}', EIGEN_ROOT='${EIGEN_ROOT}')")
    endif()
endif(WIN32)
