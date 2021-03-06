CC = g++

#EXTRA_CFLAGS= -DHAVE_SVM_STRUCT   # uncomment if linking SVM^struct 
#EXTRA_OBJ_FILES=svm_struct/*.o    # uncomment if linking SVM^struct
#EXTRA=svmstruct

INCDIR = -Iinclude -Ijson $(EXTRA_CFLAGS)
CFLAGS=$(INCDIR) -fopenmp
CFLAGS_DEBUG= -g -Wall
CFLAGS_RELEASE= -O3 -fomit-frame-pointer -ffast-math -msse3
CFLAGS_SHARED= -fPIC
LD= g++
LDFLAGS= -shared
AR=ar
ARFLAGS=rcs


CONFIGURATION=release_static

CPP_FILES := $(wildcard src/*.cpp)
HEADER_FILES := $(wildcard include/*.h) $(wildcard json/json/*.h) 
OBJ_FILES=$(addprefix obj/$(CONFIGURATION)/,$(notdir $(CPP_FILES:.cpp=.o)))

all: $(EXTRA) $(CONFIGURATION)

debug_shared: CONFIGURATION=debug_shared
debug_shared: OBJ_FILES=$(addprefix obj/debug_shared/,$(notdir $(CPP_FILES:.cpp=.o)))
debug_shared: libjson $(addprefix obj/debug_shared/,$(notdir $(CPP_FILES:.cpp=.o))) lib/libdonline_structured_svm.so example

release_shared: CONFIGURATION=release_shared
release_shared: OBJ_FILES=$(addprefix obj/release_shared/,$(notdir $(CPP_FILES:.cpp=.o)))
release_shared: libjson $(addprefix obj/release_shared/,$(notdir $(CPP_FILES:.cpp=.o))) lib/libonline_structured_svm.so example

debug_static: CONFIGURATION=debug_static
debug_static: OBJ_FILES=$(addprefix obj/debug_static/,$(notdir $(CPP_FILES:.cpp=.o)))
debug_static: libjson $(addprefix obj/debug_static/,$(notdir $(CPP_FILES:.cpp=.o))) lib/libdonline_structured_svm.a example

release_static: CONFIGURATION=release_static
release_static: OBJ_FILES=$(addprefix obj/release_static/,$(notdir $(CPP_FILES:.cpp=.o)))
release_static: libjson $(addprefix obj/release_static/,$(notdir $(CPP_FILES:.cpp=.o))) lib/libonline_structured_svm.a example

svmstruct:
	cd svm_struct; make ofiles

clean: 
	rm -f *.o obj/*/*.o lib/*
	cd json; make clean
	cd examples; make clean

doc: include/*.h src/*.cpp examples/*.h examples/*.cpp doxygen.config
	doxygen doxygen.config

libjson:
	cd json; make $(CONFIGURATION)

example: 
	cd examples; make $(CONFIGURATION)


obj/debug_static/%.o: src/%.cpp $(HEADER_FILES) 
	$(CC) $(CFLAGS) $(CFLAGS_DEBUG) -Ijson -c $< -o $@

obj/debug_shared/%.o: src/%.cpp $(HEADER_FILES) 
	$(CC) $(CFLAGS) $(CFLAGS_DEBUG) $(CFLAGS_SHARED) -Ijson -c $< -o $@

obj/release_static/%.o: src/%.cpp $(HEADER_FILES) 
	$(CC) $(CFLAGS) $(CFLAGS_RELEASE) -Ijson -c $< -o $@

obj/release_shared/%.o: src/%.cpp $(HEADER_FILES) 
	$(CC) $(CFLAGS) $(CFLAGS_RELEASE) $(CFLAGS_SHARED) -Ijson -c $< -o $@


lib/libonline_structured_svm.so: $(OBJ_FILES)  $(EXTRA_OBJ_FILES)
	$(LD) $(LDFLAGS) $(OBJ_FILES) $(EXTRA_OBJ_FILES) -o lib/libonline_structured_svm.so

lib/libonline_structured_svm.a: $(OBJ_FILES)  $(EXTRA_OBJ_FILES)
	$(AR) $(ARFLAGS) lib/libonline_structured_svm.a $(OBJ_FILES)  $(EXTRA_OBJ_FILES)

lib/libdonline_structured_svm.so: $(OBJ_FILES)  $(EXTRA_OBJ_FILES)
	$(LD) $(LDFLAGS) $(OBJ_FILES) $(EXTRA_OBJ_FILES) -o lib/libdonline_structured_svm.so

lib/libdonline_structured_svm.a: $(OBJ_FILES)  $(EXTRA_OBJ_FILES)
	$(AR) $(ARFLAGS) lib/libdonline_structured_svm.a $(OBJ_FILES) $(EXTRA_OBJ_FILES)


