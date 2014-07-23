LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

include ../OpenCV-2.3.1/share/OpenCV/OpenCV.mk

LOCAL_MODULE    := engine
LOCAL_SRC_FILES := engine.cpp
LOCAL_LDLIBS +=  -llog -ldl

include $(BUILD_SHARED_LIBRARY)
