#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_PATH
# GLEW_LIBRARIES
# GLEW_RUNTIME_LIBRARIES
#
find_path( GLEW_DIR include/GL/glew.h  )

#set(GLEW_LIBRARIES "")
#set(GLEW_RUNTIME_LIBRARIES "")

if(WIN32)

    if(MSVC)
	if(CMAKE_CL_64)
	    set(GLEWLIB_SUFFIX "/Release/x64")
	else(CMAKE_CL_64)
	    set(GLEWLIB_SUFFIX "/Release/Win32")
	endif(CMAKE_CL_64)
    endif(MSVC)

    find_path( GLEW_INCLUDE_PATH GL/glew.h PATHS ${GLEW_DIR} PATH_SUFFIXES include DOC "The directory where GL/glew.h resides")
                    
    find_library( GLEW_LIBRARY glew32 PATHS ${GLEW_DIR} ${GLEW_INCLUDE_PATH} PATH_SUFFIXES lib lib${GLEWLIB_SUFFIX} DOC "The GLEW shared library" )
    find_file( GLEW_RUNTIME_LIBRARIES glew32.dll PATHS ${GLEW_DIR} ${GLEW_INCLUDE_PATH} PATH_SUFFIXES bin bin${GLEWLIB_SUFFIX} )
    
else (WIN32)
    find_path( GLEW_INCLUDE_PATH GL/glew.h /usr/include /usr/local/include /sw/include /opt/local/include DOC "The directory where GL/glew.h resides")
    find_library( GLEW_LIBRARY NAMES GLEW glew PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib /sw/lib /opt/local/lib DOC "The GLEW library")
    set(GLEW_RUNTIME_LIBRARIES "")
endif(WIN32)
    

if(EXISTS "${GLEW_INCLUDE_PATH}")
    set(GLEW_LIBRARIES ${GLEW_LIBRARY})
    set( GLEW_FOUND true CACHE BOOL "" FORCE)
else ()
    set(GLEW_LIBRARIES "")
    set( GLEW_FOUND false CACHE BOOL "" FORCE)
endif()

mark_as_advanced( GLEW_FOUND GLEW_RUNTIME_LIBRARIES GLEW_LIBRARIES )
