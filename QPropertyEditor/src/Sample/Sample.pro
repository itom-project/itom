TEMPLATE = app

SOURCES =CustomTypes.cpp \
		 main.cpp \
		SampleApp.cpp \
        TestClass.cpp \
        TestSubClass.cpp \
        Vec3fProperty.cpp
HEADERS = CustomTypes.h \
	      SampleApp.h \
		  TestClass.h \
		  TestSubClass.h \
		  Vec3fProperty.h
FORMS = SampleApp.ui

CONFIG += debug_and_release

INCLUDEPATH+=../../include
QMAKE_LIBDIR+=../../lib
QMAKE_LIBDIR+=../../bin

DESTDIR = ../../bin
UI_DIR = .

CONFIG(debug, debug|release) {
	LIBS += -lQPropertyEditord
	TARGET = Sampled
	OBJECTS_DIR = ../../Build/Sample/Debug
	MOC_DIR = ../../Build/Sample/Debug
}

CONFIG(release, debug|release) {
	LIBS += -lQPropertyEditor
	TARGET = Sample
	OBJECTS_DIR = ../../Build/Sample/Release
	MOC_DIR = ../../Build/Sample/Release
}
