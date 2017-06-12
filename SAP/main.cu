#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "mySAP.h"

int main(){
	// Define
	static const int MAXN = 1000;
	static const int Boundary = 1000;
	static const int RadiusN = 6;
	static const float Radius[] = { 1, 2, 4, 8, 16, 32};
	static const float FPS = 50;
	static const float FrameTime = (float)1 / FPS;
	static const float SimulationTime = 20;

	// Allocation
	static Object obj[MAXN];
	static Object *cuObj;
	static FileObject fileObj[MAXN];
	static FileObject *cuFileObj;
	static int SweepDir = 0;

	FILE *fptr = fopen("log", "wb");

	// Preprocessing
	{
		srand(time(NULL));
		for (int i=0; i<MAXN; i++){
			obj[i].r = Radius[rand() % RadiusN];
			obj[i].pos[0] = (rand() % (int)(Boundary-2*obj[i].r)) + obj[i].r;	// X
			obj[i].pos[1] = Boundary - obj[i].r - (rand()%(Boundary/10));		// Y
			obj[i].pos[2] = (rand() % (int)(Boundary-2*obj[i].r)) + obj[i].r;	// Z
			obj[i].v[0] = (rand() % (Boundary/5+1)) - Boundary/10;
			obj[i].v[1] = 0;
			obj[i].v[2] = (rand() % (Boundary/5+1)) - Boundary/10;
			obj[i].isCollision = 0;
		}
		cudaMalloc( &cuObj, sizeof(Object)*MAXN);
		cudaMemcpy( cuObj, obj, sizeof(Object)*MAXN, cudaMemcpyHostToDevice);

		cudaMalloc( &cuFileObj, sizeof(FileObject)*MAXN);

		fwrite( &Boundary, sizeof(int), 1, fptr);
		fwrite( &MAXN, sizeof(int), 1, fptr);
	}

	// Simulation
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	float totalTime = 0;
	{
		while ( (totalTime / 1000) <= SimulationTime ){
			// SAP
			float partialTime = 0;
			{
				cudaEventRecord(start);
				myFindSweepDirection( cuObj, &SweepDir, MAXN); // Find Sweep Dirention
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&milliseconds, start, stop);
				totalTime += milliseconds;
				partialTime += milliseconds;
				printf("FindDir-%f\n", milliseconds);

				cudaEventRecord(start);
				mySort( cuObj, SweepDir, MAXN); // Sort
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&milliseconds, start, stop);
				totalTime += milliseconds;
				partialTime += milliseconds;
				printf("Sort-%f\n", milliseconds);

				cudaEventRecord(start);
				mySAP( cuObj, SweepDir, MAXN); // SAP
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&milliseconds, start, stop);
				totalTime += milliseconds;
				partialTime += milliseconds;
				printf("GSAP-%f\n", milliseconds);
			}
		
			myPrint( fptr, cuObj, cuFileObj, fileObj, MAXN, Boundary, partialTime/1000);
			myMoveObject( cuObj, MAXN, Boundary, FrameTime);
		}
	}

	// Free
	{
		cudaFree( cuObj);
		cudaFree( cuFileObj);
		fclose(fptr);
	}
	return 0;
}
