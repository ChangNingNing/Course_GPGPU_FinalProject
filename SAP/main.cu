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

int main(int argc, char *argv[]){
	// Define
	static const int MAXN = 2000000;
	static const int DrawN = 10000;
	static const int Boundary = 1000;
	static const int WSBound = 50000;
	static int nWS = 1;

	static const int RadiusN = 7;
	static const int SimulationTime = 1000;
	static const float Radius[] = { 1.5, 3, 6, 12, 24, 48, 96};
	static const float Proba[]  = { 6000, 9000, 9500, 9800, 9950, 9999, 10000};
//	static const float Proba[]  = { 0, 0, 0, 2500, 5000, 7500, 10000};
	static const float FPS = 50;
	static const float FrameTime = (float)1 / FPS;

	// Allocation
	static Object obj[MAXN];
	static Object *cuObj;
	static FileObject fileObj[MAXN];
	static FileObject *cuFileObj;
	static int SweepDir[3] = { 0, 1, 2}; // 0 for sweep, and 1, 2 for workspace subdivision.

	static ObjectPtr *cuWS;
	static int *cuTmp, *cuTmp2;
	static int *cuNWObj;
	static int *cuWSBound;
	FILE *fptr = fopen("log", "wb");

	int N = MAXN;
	if (argc > 1)
		N = atoi(argv[1]);

	// Preprocessing
	{
		srand(6);
		for (int i=0; i<N; i++){
			for (int j=0, _p = rand()%10000; j<RadiusN; j++){
				if (_p < Proba[j]){
					obj[i].r = Radius[j];
					break;
				}
			}

			obj[i].pos[0] = (rand() % (int)(Boundary-2*obj[i].r)) + obj[i].r;	// X
			obj[i].pos[1] = (rand() % (int)(Boundary-2*obj[i].r)) + obj[i].r;	// Y
			obj[i].pos[2] = (rand() % (int)(Boundary-2*obj[i].r)) + obj[i].r;	// Z
//			obj[i].pos[2] = Boundary - 100;										// Z

			obj[i].v[0] = (rand() % (Boundary/5+1)) - Boundary/10;
			obj[i].v[1] = 0;
			obj[i].v[2] = (rand() % (Boundary/5+1)) - Boundary/10;
			obj[i].nCollision = 0;

			obj[i].isDraw = (rand() % CeilDiv(N, DrawN)==0)? 1: 0;
			obj[i].group = 0;
		}
		nWS = CeilDiv(N, WSBound);
		nWS = nWS > 2? 2: nWS;

		cudaMalloc( &cuObj, sizeof(Object)*N);
		cudaMemcpy( cuObj, obj, sizeof(Object)*N, cudaMemcpyHostToDevice);

		cudaMalloc( &cuFileObj, sizeof(FileObject)*N);

		cudaMalloc( &cuTmp, sizeof(int)*N);
		cudaMalloc( &cuTmp2, sizeof(int)*N);

		cudaMalloc( &cuWS, sizeof(ObjectPtr)*N*((nWS+1)*(nWS+1)-nWS*nWS));
		cudaMalloc( &cuWSBound, sizeof(int)*((nWS+1)*(nWS+1)-nWS*nWS)*4);
		cudaMalloc( &cuNWObj, sizeof(int)*(nWS+1)*(nWS+1));

		fwrite( &Boundary, sizeof(int), 1, fptr);
		fwrite( &N, sizeof(int), 1, fptr);
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
				myFindSweepDirection( cuObj, SweepDir, N); // Find Sweep Dirention
				TIMER_END(rep);
				partialTime += rep;
				printf("FindDir-%f ms\n", rep);

				TIMER_START(rep);
				mySort( cuObj, SweepDir, N); // Sort
				TIMER_END(rep);
				partialTime += rep;
				printf("Sort-%f ms\n", rep);

/*
				TIMER_START(rep);
				mySAP( cuObj, SweepDir, N); // SAP
				TIMER_END(rep);
				partialTime += rep;
				printf("GSAP-%f ms\n", rep);

					// Check SAP
					cudaMemcpy( obj, cuObj, sizeof(Object)*N, cudaMemcpyDeviceToHost);
					int nC_SAP = 0;
					for (int i=0; i<N; i++) nC_SAP += obj[i].nCollision;
					//

					// Reset
					myMoveObject( cuObj, N, Boundary, 0.0f);

				TIMER_START(rep);
				mySAP_WB( cuObj, cuTmp, cuTmp2, SweepDir, N); // SAP Workload Balance
				TIMER_END(rep);
				partialTime += rep;
				printf("GSAP_WB-%f ms\n", rep);

					// Check SAP_WB
					cudaMemcpy( obj, cuObj, sizeof(Object)*N, cudaMemcpyDeviceToHost);
					int nC_SAP_WB = 0;
					for (int i=0; i<N; i++) nC_SAP_WB += obj[i].nCollision;
					//

					// Reset
					myMoveObject( cuObj, N, Boundary, 0.0f);
*/

				TIMER_START(rep);
				// Split Workspace
				myWS( cuWS, cuWSBound, cuNWObj, nWS, cuObj, cuTmp, SweepDir, N, Boundary);
				// SAP workspace subdivision
				mySAP_WS( cuObj, cuWS, cuNWObj, nWS, cuTmp, cuTmp2, SweepDir, N);
				TIMER_END(rep);
				partialTime += rep;
				printf("GSAP_WS-%f ms\n", rep);

/*
					// Check SAP_WS
					cudaMemcpy( obj, cuObj, sizeof(Object)*N, cudaMemcpyDeviceToHost);
					int nC_SAP_WS = 0;
					for (int i=0; i<N; i++) nC_SAP_WS += obj[i].nCollision;
					//

				// check
				printf("--- Check ---\n");
				printf("SAP    : %d\n", nC_SAP);
				printf("SAP_WB : %d\n", nC_SAP_WB);
				printf("SAP_WS : %d\n", nC_SAP_WS);
				assert(nC_SAP == nC_SAP_WB);
				assert(nC_SAP_WS >= nC_SAP);
				//
*/
			}
			myPrint( cuObj, cuFileObj, fileObj, cuTmp, N, partialTime/1000, fptr);
			myMoveObject( cuObj, N, Boundary, FrameTime);
		}
	}

	// Free
	{
		cudaFree( cuObj);
		cudaFree( cuFileObj);
		cudaFree( cuTmp);
		cudaFree( cuTmp2);
		cudaFree( cuWS);
		cudaFree( cuWSBound);
		cudaFree( cuNWObj);
		fclose(fptr);
	}
	return 0;
}
