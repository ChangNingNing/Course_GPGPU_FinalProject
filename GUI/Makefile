# C++ compiler
CC = g++ -std=c++11

# Compile flags
CFLAGS = -O2 -Wall -fopenmp

# Header files
HDS = -I../include

#GL flag
GL = -lglut32 -lglu32 -lopengl32 -lfreeglut

all: main

main: main.cpp
	$(CC) $(CFLAGS) $(HDS) main.cpp -o main $(GL)

clean:
	rm *.exe

