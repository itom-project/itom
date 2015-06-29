# - Find LAPACKE
# Find the native LAPACKE includes and library
# This module defines
# LAPACKE_INCLUDE_DIR, where to find fftw3.h, etc.
# LAPACKE_LIBRARIES, the libraries needed to use LAPACKE.
# LAPACKE_FOUND, If false, do not try to use LAPACKE.
# also defined, but not for general use are
# LAPACKE_LIBRARY, where to find the LAPACKE library.

SET(LAPACKE_FOUND false)

find_path(LAPACKE_DIR lapacke.h PATH_SUFFIXES "include" DOC "Root directory of lapacke")
FIND_PATH(LAPACKE_INCLUDE_DIR lapacke.h PATHS ${LAPACKE_DIR} PATH_SUFFIXES "include")

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
	
set(LAPACKE_COMPONENTS blas lapack lapacke gfortran-3 gcc_s_seh-1 quadmath-0 tmglib winpthread-1) #blas must be first, since it should be added first to the linker
SET(LAPACKE_FOUND true)
## Loop over each components
foreach(__LIB ${LAPACKE_COMPONENTS})
		
		FIND_LIBRARY(LAPACKE_${__LIB}_LIBRARY NAMES lib${__LIB} ${__LIB} PATHS ${LAPACKE_DIR}/lib/${LAPACKE_LIBSUFFIX} NO_DEFAULT_PATH)
		
		#Add to the general list
		if(LAPACKE_${__LIB}_LIBRARY)
			set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARIES} ${LAPACKE_${__LIB}_LIBRARY})
		endif(LAPACKE_${__LIB}_LIBRARY)
		
		IF(WIN32)
			FIND_FILE(LAPACKE_${__LIB}_RUNTIME NAMES lib${__LIB}.dll ${__LIB}.dll PATHS ${LAPACKE_DIR}/bin/${LAPACKE_LIBSUFFIX} NO_DEFAULT_PATH)
			
			IF(LAPACKE_${__LIB}_RUNTIME)
				SET(LAPACKE_RUNTIME_LIBRARIES ${LAPACKE_RUNTIME_LIBRARIES} ${LAPACKE_${__LIB}_RUNTIME})
			ENDIF(LAPACKE_${__LIB}_RUNTIME)
		ENDIF(WIN32)
endforeach(__LIB)

IF(LAPACKE_lapacke_LIBRARY)
ELSE()
	SET(LAPACKE_FOUND false)
ENDIF()
