INCLUDES=../utils
CXXFLAGS=-std=c++11 -O2 -arch sm_30 -lineinfo

all: main

main: main.o mySAP.o 
	nvcc $(CXXFLAGS) main.o mySAP.o -o main

main.o: main.cu
	nvcc $(CXXFLAGS) main.cu -c

mySAP.o: mySAP.cu
	nvcc $(CXXFLAGS) mySAP.cu -c

clean:
	rm main *.o
