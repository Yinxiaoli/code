# Makefile for empty SVM-struct API, 03.10.06

#Call 'make' using the following line to make CYGWIN produce stand-alone Windows executables
#		make 'SFLAGS=-mno-cygwin'

#Use the following to compile under unix or cygwin
CC = g++
LD = g++

INCLUDES = -I../../include -I../../online_structured_svm/include -I../../online_structured_svm/json -I/usr/include/opencv -I/usr/local/include/opencv -fopenmp -DUSE_OPENMP
LIBS = -L. -lsvm_struct -L../../lib -lvisipedia -L../lib -lonline_structured_svm -L../json/lib -ljson  -lcv -lcxcore -lhighgui -fopenmp
#CFLAGS =   $(SFLAGS) -O3 -fomit-frame-pointer -ffast-math -Wall $(INCLUDES)
CFLAGS =  $(SFLAGS) -g -Wall $(INCLUDES)
AR=ar
ARFLAGS=rcs

OFILES = svm_struct_api.o svm_struct_learn_custom.o svm_struct_classify.o svm_struct_main.o svm_struct_common.o svm_struct_learn.o svm_hideo.o svm_learn.o svm_common.o 

all: svm_empty_learn svm_empty_classify

.PHONY: clean
clean: 
	rm -f *.o *.tcov *.d core gmon.out *.stackdump libsvm_struct.a svm_empty_classify svm_empty_learn

ofiles: svm_struct_api.o svm_struct_learn_custom.o svm_struct_classify.o svm_struct_main.o svm_struct_common.o svm_struct_learn.o svm_hideo.o svm_learn.o svm_common.o 

#-------------------------#
#----  SVM empty API  ----#
#-------------------------#

libsvm_struct.a: $(OFILES)
	$(AR) $(ARFLAGS) libsvm_struct.a $(OFILES)

svm_empty_classify: libsvm_struct.a classify_main.cpp
	$(CC) $(CFLAGS) classify_main.cpp -o svm_empty_classify $(LIBS)

svm_empty_learn: libsvm_struct.a learn_main.cpp 
	$(CC) $(CFLAGS) learn_main.cpp -o svm_empty_learn  $(LIBS)




svm_struct_api.o: svm_struct_api.cpp svm_struct_api.h svm_struct_api_types.h svm_struct/svm_struct_common.h
	$(CC) -c $(CFLAGS) svm_struct_api.cpp -o svm_struct_api.o

svm_struct_learn_custom.o: svm_struct_learn_custom.cpp svm_struct_api.h svm_light/svm_common.h svm_struct_api_types.h svm_struct/svm_struct_common.h
	$(CC) -c $(CFLAGS) svm_struct_learn_custom.cpp -o svm_struct_learn_custom.o




svm_hideo.o: svm_light/svm_hideo.cpp
	$(CC) -c $(CFLAGS) svm_light/svm_hideo.cpp -o svm_hideo.o 

svm_common.o: svm_light/svm_common.cpp svm_light/svm_common.h svm_light/kernel.h
	$(CC) -c $(CFLAGS) svm_light/svm_common.cpp -o svm_common.o 

svm_learn.o: svm_light/svm_learn.cpp svm_light/svm_common.h
	$(CC) -c $(CFLAGS) svm_light/svm_learn.cpp -o svm_learn.o 

svm_learn_main.o: svm_light/svm_learn_main.cpp svm_light/svm_learn.h svm_light/svm_common.h
	$(CC) -c $(CFLAGS) svm_light/svm_learn_main.cpp -o svm_learn_main.o 

svm_classify.o: svm_light/svm_classify.cpp svm_light/svm_common.h svm_light/kernel.h
	$(CC) -c $(CFLAGS) svm_light/svm_classify.cpp -o svm_classify.o



svm_struct_common.o: svm_struct/svm_struct_common.cpp svm_struct/svm_struct_common.h svm_struct_api_types.h
	$(CC) -c $(CFLAGS) svm_struct/svm_struct_common.cpp -o svm_struct_common.o

svm_struct_learn.o: svm_struct/svm_struct_learn.cpp svm_light/svm_learn.h svm_struct/svm_struct_common.h svm_struct_api.h svm_struct_api_types.h
	$(CC) -c $(CFLAGS) svm_struct/svm_struct_learn.cpp -o svm_struct_learn.o

svm_struct_main.o: svm_struct/svm_struct_main.cpp svm_light/svm_common.h svm_light/svm_learn.h svm_struct/svm_struct_learn.h svm_struct/svm_struct_common.h svm_struct_api.h svm_struct_api_types.h
	$(CC) -c $(CFLAGS) svm_struct/svm_struct_main.cpp -o svm_struct_main.o

svm_struct_classify.o: svm_struct/svm_struct_classify.cpp svm_struct/svm_struct_common.h svm_struct_api_types.h svm_struct_api.h svm_light/svm_common.h 
	$(CC) -c $(CFLAGS) svm_struct/svm_struct_classify.cpp -o svm_struct_classify.o
