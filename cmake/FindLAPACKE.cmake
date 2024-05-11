# - Find LAPACKE
# Find the native LAPACKE includes and library
# This module defines
# LAPACKE_INCLUDE_DIR, where to find fftw3.h, etc.
# LAPACKE_LIBRARIES, the libraries needed to use LAPACKE.
# LAPACKE_FOUND, If false, do not try to use LAPACKE.
# also defined, but not for general use are
# LAPACKE_LIBRARY, where to find the LAPACKE library.

set(LAPACKE_FOUND false)

if(NOT EXISTS ${LAPACKE_DIR})
    if(EXISTS $ENV{LAPACKE_ROOT})
        set(LAPACKE_DIR $ENV{LAPACKE_ROOT} CACHE PATH "Root directory of LAPACKE, containing sub-directories library and include")
    else(EXISTS $ENV{LAPACKE_ROOT})
        set(LAPACKE_DIR "LAPACKE_DIR-NOTFOUND" CACHE PATH "Root directory of LAPACKE, containing sub-directories library and include")
    endif(EXISTS $ENV{LAPACKE_ROOT})
endif(NOT EXISTS ${LAPACKE_DIR})

message(STATUS "LAPACKE_DIR: ${LAPACKE_DIR}")

#find_path(LAPACKE_DIR lapacke.h PATH_SUFFIXES "include" DOC "Root directory of lapacke")
find_path(LAPACKE_INCLUDE_DIR lapacke.h PATHS ${LAPACKE_DIR} PATH_SUFFIXES "include")

if(MSVC)
	if(CMAKE_CL_64)
		set(LAPACKE_LIBSUFFIX "/x64")
	else(CMAKE_CL_64)
		set(LAPACKE_LIBSUFFIX "/x86")
	endif(CMAKE_CL_64)
endif(MSVC)

## Initiate the variable before the loop
set(LAPACKE_LIBRARIES "")
#set(LAPACKE_RUNTIME_LIBRARIES "")

#Remove the cache value
set(LAPACKE_RUNTIME_LIBRARIES "" CACHE STRING "" FORCE)

set(LAPACKE_COMPONENTS blas lapack lapacke gfortran-3 gcc_s_seh-1 gcc_s_dw2-1 quadmath-0 tmglib winpthread-1) #blas must be first, since it should be added first to the linker
set(LAPACKE_FOUND true)
## Loop over each components
foreach(__LIB ${LAPACKE_COMPONENTS})

		find_library(LAPACKE_${__LIB}_LIBRARY NAMES lib${__LIB} ${__LIB} PATHS ${LAPACKE_DIR}/lib/${LAPACKE_LIBSUFFIX} NO_DEFAULT_PATH)

		#Add to the general list
		if(LAPACKE_${__LIB}_LIBRARY)
			set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARIES} ${LAPACKE_${__LIB}_LIBRARY})
		endif(LAPACKE_${__LIB}_LIBRARY)

		if(WIN32)
			find_file(LAPACKE_${__LIB}_RUNTIME NAMES lib${__LIB}.dll ${__LIB}.dll PATHS ${LAPACKE_DIR}/bin/${LAPACKE_LIBSUFFIX} NO_DEFAULT_PATH)

			if(LAPACKE_${__LIB}_RUNTIME)
				set(LAPACKE_RUNTIME_LIBRARIES ${LAPACKE_RUNTIME_LIBRARIES} ${LAPACKE_${__LIB}_RUNTIME})
				message(STATUS "LAPACKE_RUNTIME_LIBRARIES: ${LAPACKE_RUNTIME_LIBRARIES}")
			endif(LAPACKE_${__LIB}_RUNTIME)

			message(STATUS "LAPACKE_${__LIB}_RUNTIME: ${LAPACKE_${__LIB}_RUNTIME}")
		endif(WIN32)
endforeach(__LIB)

if(LAPACKE_lapacke_LIBRARY)
else()
	set(LAPACKE_FOUND false)
endif()
