INCLUDES=../utils
CXXFLAGS=-std=c++11 -O2 -arch sm_30

all: main

main: main.cu mySAP.cu
	nvcc $(CXXFLAGS) main.cu mySAP.cu -o main

clean:
	rm main
