CC=g++
#CFLAGS= -Ilocal/include -Ilocal/include/opencv -Iinclude -Ionline_structured_svm/include -Ionline_structured_svm/json -Ionline_structured_svm/examples -fopenmp -DUSE_OPENMP
CFLAGS= -Ilocal/include -I/usr/include/opencv -I/usr/local/include/opencv -Iinclude -Ionline_structured_svm/include -Ionline_structured_svm/json -Ionline_structured_svm/examples -fopenmp -DUSE_OPENMP
CFLAGS_DEBUG= -g -Wall
CFLAGS_RELEASE= -O3 -fomit-frame-pointer -ffast-math -msse3
CFLAGS_SHARED= -fPIC
LD=g++
LDFLAGS= -shared
AR=ar
ARFLAGS=rcs

include Makefile.inc
ifdef MATLAB_DIR
	CFLAGS := $(CFLAGS) -DHAVE_MATLAB -I$(MATLAB_DIR)/extern/include/ $(EXTRA_INCLUDE)
endif


CONFIGURATION=release_static

CPP_FILES := $(wildcard src/*.cpp)
HEADER_FILES := $(wildcard include/*.h) $(wildcard online_structured_svm/include/*.h) $(wildcard online_structured_svm/json/json/*.h) $(wildcard online_structured_svm/json/json/*.h) $(wildcard online_structured_svm/examples/*.h)
OBJ_FILES_EXTRA=obj/$(CONFIGURATION)/structured_svm_multiclass.o
OBJ_FILES=$(addprefix obj/$(CONFIGURATION)/,$(notdir $(CPP_FILES:.cpp=.o))) $(OBJ_FILES_EXTRA)

all: $(CONFIGURATION)

debug_shared: CONFIGURATION=debug_shared
debug_shared: OBJ_FILES_EXTRA=obj/$(CONFIGURATION)/structured_svm_multiclass.o
debug_shared: OBJ_FILES=$(addprefix obj/debug_shared/,$(notdir $(CPP_FILES:.cpp=.o))) $(OBJ_FILES_EXTRA)
debug_shared: svm $(addprefix obj/debug_shared/,$(notdir $(CPP_FILES:.cpp=.o))) lib/libdvisipedia.so example

release_shared: CONFIGURATION=release_shared
release_shared: OBJ_FILES_EXTRA=obj/$(CONFIGURATION)/structured_svm_multiclass.o
release_shared: OBJ_FILES=$(addprefix obj/release_shared/,$(notdir $(CPP_FILES:.cpp=.o))) $(OBJ_FILES_EXTRA)
release_shared: svm $(addprefix obj/release_shared/,$(notdir $(CPP_FILES:.cpp=.o))) lib/libvisipedia.so example

debug_static: CONFIGURATION=debug_static
debug_static: OBJ_FILES_EXTRA=obj/$(CONFIGURATION)/structured_svm_multiclass.o
debug_static: OBJ_FILES=$(addprefix obj/debug_static/,$(notdir $(CPP_FILES:.cpp=.o))) $(OBJ_FILES_EXTRA)
debug_static: svm $(addprefix obj/debug_static/,$(notdir $(CPP_FILES:.cpp=.o))) lib/libdvisipedia.a example

release_static: CONFIGURATION=release_static
release_static: OBJ_FILES_EXTRA=obj/$(CONFIGURATION)/structured_svm_multiclass.o
release_static: OBJ_FILES=$(addprefix obj/release_static/,$(notdir $(CPP_FILES:.cpp=.o))) $(OBJ_FILES_EXTRA)
release_static: svm $(addprefix obj/release_static/,$(notdir $(CPP_FILES:.cpp=.o))) lib/libvisipedia.a example


obj: $(OBJ_FILES) 

svm:
	cd online_structured_svm; make $(CONFIGURATION)

example: online_structured_svm
	cd examples; make $(CONFIGURATION)

doc: include/*.h examples/*.cpp online_structured_svm/include/*.h online_structured_svm/examples/*.cpp doxygen.config
	doxygen doxygen.config

clean: 
	rm -f obj/*/*.o lib/*.so lib/*.a
	cd online_structured_svm; make clean
	cd examples; make clean


lib/libvisipedia.so: $(OBJ_FILES) obj/release_shared/structured_svm_multiclass.o
	$(LD) $(LDFLAGS) $(OBJ_FILES) -o lib/libvisipedia.so 

lib/libvisipedia.a: $(OBJ_FILES) obj/release_static/structured_svm_multiclass.o
	$(AR) $(ARFLAGS) lib/libvisipedia.a $(OBJ_FILES)

lib/libdvisipedia.so: $(OBJ_FILES) obj/debug_shared/structured_svm_multiclass.o
	$(LD) $(LDFLAGS) $(OBJ_FILES) -o lib/libdvisipedia.so 

lib/libdvisipedia.a: $(OBJ_FILES) obj/debug_static/structured_svm_multiclass.o
	$(AR) $(ARFLAGS) lib/libdvisipedia.a $(OBJ_FILES)


obj/debug_static/structured_svm_multiclass.o: online_structured_svm/examples/structured_svm_multiclass.cpp $(HEADER_FILES) 
	$(CC) $(CFLAGS) $(CFLAGS_DEBUG) -DNO_SERVER -c online_structured_svm/examples/structured_svm_multiclass.cpp -o obj/debug_static/structured_svm_multiclass.o

obj/debug_shared/structured_svm_multiclass.o: online_structured_svm/examples/structured_svm_multiclass.cpp $(HEADER_FILES) 
	$(CC) $(CFLAGS) $(CFLAGS_DEBUG) $(CFLAGS_SHARED) -DNO_SERVER -c online_structured_svm/examples/structured_svm_multiclass.cpp -o obj/debug_shared/structured_svm_multiclass.o

obj/release_static/structured_svm_multiclass.o: online_structured_svm/examples/structured_svm_multiclass.cpp $(HEADER_FILES) 
	$(CC) $(CFLAGS) $(CFLAGS_RELEASE) -DNO_SERVER -c online_structured_svm/examples/structured_svm_multiclass.cpp -o obj/release_static/structured_svm_multiclass.o

obj/release_shared/structured_svm_multiclass.o: online_structured_svm/examples/structured_svm_multiclass.cpp $(HEADER_FILES) 
	$(CC) $(CFLAGS) $(CFLAGS_RELEASE) $(CFLAGS_SHARED) -DNO_SERVER -c online_structured_svm/examples/structured_svm_multiclass.cpp -o obj/release_shared/structured_svm_multiclass.o



obj/debug_static/%.o: src/%.cpp $(HEADER_FILES) 
	$(CC) $(CFLAGS) $(CFLAGS_DEBUG) -c $< -o $@

obj/debug_shared/%.o: src/%.cpp $(HEADER_FILES) 
	$(CC) $(CFLAGS) $(CFLAGS_DEBUG) $(CFLAGS_SHARED) -c $< -o $@

obj/release_static/%.o: src/%.cpp $(HEADER_FILES) 
	$(CC) $(CFLAGS) $(CFLAGS_RELEASE) -c $< -o $@

obj/release_shared/%.o: src/%.cpp $(HEADER_FILES) 
	$(CC) $(CFLAGS) $(CFLAGS_RELEASE)  $(CFLAGS_SHARED) -c $< -o $@

