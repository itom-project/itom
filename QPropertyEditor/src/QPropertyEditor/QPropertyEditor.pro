TEMPLATE = lib
CONFIG += staticlib
CONFIG += debug_and_release

SOURCES = ColorCombo.cpp \
          Property.cpp \
          QPropertyEditorWidget.cpp \
          QPropertyModel.cpp \
          QVariantDelegate.cpp \
          EnumProperty.cpp

HEADERS=ColorCombo.h \
        Property.h \
        QPropertyEditorWidget.h \
        QPropertyModel.h \
        QVariantDelegate.h \
        EnumProperty.h

INCLUDEPATH+=../../include
DESTDIR = ../../lib
UI_DIR = .

CONFIG(debug, debug|release) {
        TARGET = QPropertyEditord
        OBJECTS_DIR = ../../Build/QPropertyEditor/Debug
        MOC_DIR = ../../Build/QPropertyEditor/Debug
}

CONFIG(release, debug|release) {

        TARGET = QPropertyEditor
        OBJECTS_DIR = ../../Build/QPropertyEditor/Release
        MOC_DIR = ../../Build/QPropertyEditor/Release
        DEFINES+=QT_NO_DEBUG
}

macx {
        CONFIG += x86 ppc
        QMAKE_MAC_SDK = /Developer/SDKs/MacOSX10.4u.sdk
        QMAKE_LFLAGS += -mmacosx-version-min=10.4
}

unix {
        system(mkdir -p ../../include/QPropertyEditor)
        system(cp *.h ../../include/QPropertyEditor)
}

win32 {
        system(xcopy /y *.h ../../include/QPropertyEditor/ )
}
