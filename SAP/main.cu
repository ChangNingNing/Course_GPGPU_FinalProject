#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>
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
	static const int MAXN = 1500000;
	static const int DrawN = 1000;
	static const int Boundary = 1000;
	static const int WSBound = 100000;
	static int nWS = 1;

	static const int RadiusN = 7;
	static const int SimulationTime = 5;
	static const float Radius[] = { 1.5, 3, 6, 12, 24, 48, 96};
	static const float Proba[]  = { 6000, 9000, 9500, 9800, 9950, 9999, 10000};
	static const float FPS = 50;
	static const float FrameTime = (float)1 / FPS;

	// Allocation
	static Object obj[MAXN];
	static Object *cuObj;
	static FileObject fileObj[MAXN];
	static FileObject *cuFileObj;
	static int SweepDir[2] = { 0, 2}; // 0 for sweep, and 1 for workspace subdivision.

	static Object **cuWS;
	static int *cuTmp, *cuTmp2;
	static int *cuNWObj;
	FILE *fptr = fopen("log", "wb");

	// Preprocessing
	{
		srand(time(NULL));
		for (int i=0; i<MAXN; i++){
			for (int j=0, _p = rand()%10000; j<RadiusN; j++){
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

			obj[i].isDraw = (rand() % CeilDiv(MAXN, DrawN)==0)? 1: 0;
			obj[i].group = 0;
		}
		nWS = CeilDiv(MAXN, WSBound);
		nWS = nWS > 2? 2: nWS;

		cudaMalloc( &cuObj, sizeof(Object)*MAXN);
		cudaMemcpy( cuObj, obj, sizeof(Object)*MAXN, cudaMemcpyHostToDevice);

		cudaMalloc( &cuFileObj, sizeof(FileObject)*MAXN);

		cudaMalloc( &cuTmp, sizeof(int)*MAXN);
		cudaMalloc( &cuTmp2, sizeof(int)*MAXN);

		cudaMalloc( &cuWS, sizeof(Object *)*MAXN*(nWS-1));
		cudaMalloc( &cuNWObj, sizeof(int)*nWS*4);

		fwrite( &Boundary, sizeof(int), 1, fptr);
		fwrite( &MAXN, sizeof(int), 1, fptr);
	}

	// Simulation
	float rep;
	TIMER_CREATE(rep);
	int totalTime = 0;
	{
		while ( totalTime <= SimulationTime ){
			printf("\n%d\n", totalTime);
			totalTime++;
			// SAP
			float partialTime = 0;
			{
				TIMER_START(rep);
				myFindSweepDirection( cuObj, SweepDir, MAXN); // Find Sweep Dirention
				TIMER_END(rep);
				partialTime += rep;
				printf("FindDir-%f ms\n", rep);

				TIMER_START(rep);
				mySort( cuObj, SweepDir, MAXN); // Sort
				TIMER_END(rep);
				partialTime += rep;
				printf("Sort-%f ms\n", rep);

				TIMER_START(rep);
				mySAP( cuObj, SweepDir, MAXN); // SAP
				TIMER_END(rep);
				partialTime += rep;
				printf("GSAP-%f ms\n", rep);

					// Check SAP
					cudaMemcpy( obj, cuObj, sizeof(Object)*MAXN, cudaMemcpyDeviceToHost);
					int nP_SAP = 0;
					for (int i=0; i<MAXN; i++) nP_SAP += obj[i].isCollision;
					//

					// Reset
					myMoveObject( cuObj, MAXN, Boundary, 0.0f);

				TIMER_START(rep);
				mySAP_WB( cuObj, cuTmp, cuTmp2, SweepDir, MAXN); // SAP Workload Balance
				TIMER_END(rep);
				partialTime += rep;
				printf("GSAP_WB-%f ms\n", rep);

					// Check SAP_WB
					cudaMemcpy( obj, cuObj, sizeof(Object)*MAXN, cudaMemcpyDeviceToHost);
					int nP_SAP_WB = 0;
					for (int i=0; i<MAXN; i++) nP_SAP_WB += obj[i].isCollision;
					//

					// Reset
					myMoveObject( cuObj, MAXN, Boundary, 0.0f);

				TIMER_START(rep);
				// Split Workspace
				myWS( cuWS, cuNWObj, nWS, cuObj, cuTmp, SweepDir, MAXN, Boundary);
				TIMER_END(rep);
				partialTime += rep;
				printf("Workspace-%f ms\n", rep);

				TIMER_START(rep);
				// SAP workspace subdivision
				mySAP_WS( cuObj, cuWS, cuNWObj, nWS, cuTmp, cuTmp2, SweepDir, MAXN);
				TIMER_END(rep);
				partialTime += rep;
				printf("GSAP_WS-%f ms\n", rep);

					// Check SAP_WS
					cudaMemcpy( obj, cuObj, sizeof(Object)*MAXN, cudaMemcpyDeviceToHost);
					int nP_SAP_WS = 0;
					for (int i=0; i<MAXN; i++) nP_SAP_WS += obj[i].isCollision;
					//

				// check
//				printf("--- Check ---\n");
//				printf("SAP    : %d\n", nP_SAP);
//				printf("SAP_WB : %d\n", nP_SAP_WB);
//				printf("SAP_WS : %d\n", nP_SAP_WS);
				assert(nP_SAP == nP_SAP_WB);
				assert(nP_SAP_WS >= nP_SAP);
				//
			}
			myPrint( cuObj, cuFileObj, fileObj, cuTmp, MAXN, partialTime/1000, fptr);
			myMoveObject( cuObj, MAXN, Boundary, FrameTime);
		}
	}

	// Free
	{
		cudaFree( cuObj);
		cudaFree( cuFileObj);
		cudaFree( cuTmp);
		cudaFree( cuTmp2);
		cudaFree( cuWS);
		cudaFree( cuNWObj);
		fclose(fptr);
	}
	return 0;
}
