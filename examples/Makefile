INCDIR_CV = -I/usr/include/opencv -I/usr/local/include/opencv -I../local/include/opencv -I../local/include
#LIBS_CV = -lcv -lcxcore -lcvaux -lhighgui -fopenmp # opencv-1.0
LIBS_CV = -lopencv_highgui -lopencv_imgproc -lopencv_core # opencv-2.0

CC=g++
CFLAGS= $(INCDIR_CV) -I../include -I../online_structured_svm/examples -I../online_structured_svm/include -I../online_structured_svm/json -fopenmp -DUSE_OPENMP
CFLAGS_DEBUG= -g -Wall
CFLAGS_RELEASE= -O3 -fomit-frame-pointer -ffast-math -msse3
CFLAGS_SHARED= -fPIC
LIBS= -L../lib -L../online_structured_svm/lib -L../online_structured_svm/json/lib -lm -L../local/lib 
LIBS_RELEASE=-lvisipedia -lonline_structured_svm -ljson
LIBS_DEBUG=-ldvisipedia -ldonline_structured_svm -ldjson
LD=g++
LDFLAGS= -shared
AR=ar
ARFLAGS=rcs


include ../Makefile.inc
ifdef MATLAB_DIR
	CFLAGS := $(CFLAGS) -DHAVE_MATLAB -I$(MATLAB_DIR)/extern/include/
	LIBS_CV := $(LIBS_CV) -L$(MATLAB_DIR)/bin/glnxa64 -lmx -lmat -leng 
	LD_LIBRARY_PATH= .:$LD_LIBRARY_PATH:$(MATLAB_DIR)/bin/glnxa64
endif

CONFIGURATION=release_static

CPP_FILES := $(wildcard *.cpp)
HEADER_FILES = ../include/*.h ../online_structured_svm/include/*.h ../online_structured_svm/json/json/*.h 
OUT_FILES := $(addprefix bin/$(CONFIGURATION)/,$(notdir $(CPP_FILES:.cpp=.out)))


all: $(CONFIGURATION)

debug_shared: CONFIGURATION=debug_shared
debug_shared: OUT_FILES=$(addprefix bin/debug_shared/,$(notdir $(CPP_FILES:.cpp=.out)))
debug_shared: LIBS += $(LIBS_DEBUG) $(LIBS_CV)
debug_shared: $(addprefix bin/debug_shared/,$(notdir $(CPP_FILES:.cpp=.out)))

release_shared: CONFIGURATION=release_shared
release_shared: OUT_FILES=$(addprefix bin/release_shared/,$(notdir $(CPP_FILES:.cpp=.out)))
release_shared: LIBS += $(LIBS_RELEASE) $(LIBS_CV)
release_shared: $(addprefix bin/release_shared/,$(notdir $(CPP_FILES:.cpp=.out)))

debug_static: CONFIGURATION=debug_static
debug_static: OUT_FILES=$(addprefix bin/debug_static/,$(notdir $(CPP_FILES:.cpp=.out)))
debug_static: LIBS += $(LIBS_DEBUG) $(LIBS_CV)
debug_static: $(addprefix bin/debug_static/,$(notdir $(CPP_FILES:.cpp=.out)))

release_static: CONFIGURATION=release_static
release_static: OUT_FILES=$(addprefix bin/release_static/,$(notdir $(CPP_FILES:.cpp=.out)))
release_static: LIBS += $(LIBS_RELEASE) $(LIBS_CV)
release_static: $(addprefix bin/release_static/,$(notdir $(CPP_FILES:.cpp=.out)))


clean: 
	rm -f bin/*/*.out

bin/debug_static/%.out: %.cpp $(HEADER_FILES) ../lib/libdvisipedia.a ../online_structured_svm/lib/libdonline_structured_svm.a ../online_structured_svm/json/lib/libdjson.a
	$(CC) $(CFLAGS) $(CFLAGS_DEBUG) $<  -o $@ $(LIBS)

bin/debug_shared/%.out: %.cpp $(HEADER_FILES)
	$(CC) $(CFLAGS) $(CFLAGS_DEBUG) $(CFLAGS_SHARED) $< -o $@ $(LIBS)

bin/release_static/%.out: %.cpp $(HEADER_FILES) ../lib/libvisipedia.a ../online_structured_svm/lib/libonline_structured_svm.a ../online_structured_svm/json/lib/libjson.a
	$(CC) $(CFLAGS) $(CFLAGS_RELEASE) $< -o $@ $(LIBS)

bin/release_shared/%.out: %.cpp $(HEADER_FILES)
	$(CC) $(CFLAGS) $(CFLAGS_RELEASE) $(CFLAGS_SHARED)  $< -o $@ $(LIBS)

