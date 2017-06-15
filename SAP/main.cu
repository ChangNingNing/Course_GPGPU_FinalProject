#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "mySAP.h"
#define TIMER_CREATE(t)                      \
  cudaEvent_t t##_start, t##_end;            \
  cudaEventCreate(&t##_start);               \
  cudaEventCreate(&t##_end);               
 
 
#define TIMER_START(t)                          \
  cudaEventRecord(t##_start);                   \
  cudaEventSynchronize(t##_start);              \
 
 
#define TIMER_END(t)                                          \
  cudaEventRecord(t##_end);                                   \
  cudaEventSynchronize(t##_end);                              \
  cudaEventElapsedTime(&t, t##_start, t##_end);               
 
int main(){
	// Define
	static const int MAXN = 102000;
	static const int DrawN = 1000;
	static const int Boundary = 1000;
	static const int RadiusN = 7;
	static const float Radius[] = { 3, 6, 12, 24, 48, 96, 192};
	static const float Proba[]  = { 700, 900, 950, 990, 998, 1000};
	static const float FPS = 50;
	static const float FrameTime = (float)1 / FPS;
	static const float SimulationTime = 60;

	// Allocation
	static Object obj[MAXN];
	static Object *cuObj;
	static FileObject fileObj[MAXN];
	static FileObject *cuFileObj;
	static int SweepDir = 0;

	static int *cuR, *cuNT;

	FILE *fptr = fopen("log", "wb");

	// Preprocessing
	{
		srand(time(NULL));
		for (int i=0; i<MAXN; i++){
			for (int j=0, _p = rand()%1000; j<RadiusN; j++){
				if (_p < Proba[j]){
					obj[i].r = Radius[j];
					break;
				}
			}

			obj[i].pos[0] = (rand() % (int)(Boundary-2*obj[i].r)) + obj[i].r;	// X
			obj[i].pos[1] = Boundary - obj[i].r - (rand()%(Boundary/10));		// Y
			obj[i].pos[2] = (rand() % (int)(Boundary-2*obj[i].r)) + obj[i].r;	// Z
			obj[i].v[0] = (rand() % (Boundary/5+1)) - Boundary/10;
			obj[i].v[1] = 0;
			obj[i].v[2] = (rand() % (Boundary/5+1)) - Boundary/10;
			obj[i].isCollision = 0;

			obj[i].isDraw = (rand() % (MAXN / DrawN)==0)? 1: 0;
		}
		cudaMalloc( &cuObj, sizeof(Object)*MAXN);
		cudaMemcpy( cuObj, obj, sizeof(Object)*MAXN, cudaMemcpyHostToDevice);

		cudaMalloc( &cuFileObj, sizeof(FileObject)*MAXN);

		cudaMalloc( &cuR, sizeof(int)*MAXN);
		cudaMalloc( &cuNT, sizeof(int)*MAXN);

		fwrite( &Boundary, sizeof(int), 1, fptr);
		fwrite( &MAXN, sizeof(int), 1, fptr);
	}

	// Simulation
	float rep;
	TIMER_CREATE(rep);
	float totalTime = 0;
	{
		while ( (totalTime / 1000) <= SimulationTime ){
			// SAP
			float partialTime = 0;
			{
				TIMER_START(rep);
				myFindSweepDirection( cuObj, &SweepDir, MAXN); // Find Sweep Dirention
				TIMER_END(rep);
				totalTime += rep;
				partialTime += rep;
				printf("FindDir-%f ms\n", rep);

				TIMER_START(rep);
				mySort( cuObj, SweepDir, MAXN); // Sort
				TIMER_END(rep);
				totalTime += rep;
				partialTime += rep;
				printf("Sort-%f ms\n", rep);
/*
				TIMER_START(rep);
				mySAP( cuObj, SweepDir, MAXN); // SAP
				TIMER_END(rep);
				totalTime += rep;
				partialTime += rep;
				printf("GSAP-%f ms\n", rep);

				// Check SAP
				cudaMemcpy( obj, cuObj, sizeof(Object)*MAXN, cudaMemcpyDeviceToHost);
				for (int i=0; i<40; i++) printf("%d ", obj[i].isCollision);
				puts("");
				for (int i=MAXN-40; i<MAXN; i++) printf("%d ", obj[i].isCollision);
				puts("");
				//
*/
				TIMER_START(rep);
				mySAP2( cuObj, cuR, cuNT, SweepDir, MAXN); // SAP
				TIMER_END(rep);
				totalTime += rep;
				partialTime += rep;
				printf("GSAP2-%f ms\n", rep);

				// Check SAP2
				cudaMemcpy( obj, cuObj, sizeof(Object)*MAXN, cudaMemcpyDeviceToHost);
				for (int i=0; i<40; i++) printf("%d ", obj[i].isCollision);
				puts("");
				for (int i=MAXN-40; i<MAXN; i++) printf("%d ", obj[i].isCollision);
				puts("");
				//
			}
		
			myPrint( cuObj, cuFileObj, fileObj, cuR, MAXN, partialTime/1000, fptr);
			myMoveObject( cuObj, MAXN, Boundary, FrameTime);
		}
	}

	// Free
	{
		cudaFree( cuObj);
		cudaFree( cuFileObj);
		cudaFree( cuR);
		cudaFree( cuNT);
		fclose(fptr);
	}
	return 0;
}
