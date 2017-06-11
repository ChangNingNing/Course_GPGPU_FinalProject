#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "mySAP.h"

//#define DEBUG

void myPrint( int Boundary, int N, Object *obj, float frameTime){
/**/
	for (int i=0; i<N; i++){
		printf("%.3f %.3f %.3f %.0f %d ", obj[i].pos[0], obj[i].pos[1], obj[i].pos[2], obj[i].r, obj[i].isCollision);
	}
/**/
	printf("%lf\n", frameTime);
}

int main(){
	// Define
	static const int MAXN = 1000000;
	static const int Boundary = 100000;
	static const int RadiusN = 6;
	static const float Radius[] = { 1, 2, 4, 8, 16, 32};
	static const float FPS = 50;
	static const float FrameTime = (float)1 / FPS;
	static const float SimulationTime = 5;

	// Allocation
	static Object obj[MAXN];
	static Object *cuObj;
	static int SweepDir[3] = { 1, 0, 0};
	static int *cuSweepDir;

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

		cudaMalloc( &cuSweepDir, sizeof(int)*3);
		cudaMemcpy( cuSweepDir, SweepDir, sizeof(int)*3, cudaMemcpyHostToDevice);

		printf("%d %d\n", Boundary, MAXN);
	}

	#ifdef DEBUG
	{
		printf("FrameTime = %lfs\n", FrameTime);
	}
	#endif

	// Simulation
	clock_t begin = clock();
	{
		while ( ((float)(clock()-begin) / CLOCKS_PER_SEC) <= SimulationTime ){
			clock_t duration = clock();
			// SAP
			{
				myFindSweepDirection( cuObj, cuSweepDir, MAXN);
				mySort( cuObj, cuSweepDir, MAXN);
				mySAP( cuObj, cuSweepDir, MAXN);
			}
			duration = clock() - duration;

			cudaMemcpy( obj, cuObj, sizeof(Object)*MAXN, cudaMemcpyDeviceToHost);
			myPrint( Boundary, MAXN, obj, (float)duration/CLOCKS_PER_SEC);

			myMoveObject( cuObj, MAXN, Boundary, FrameTime);
		}
	}

	// Free
	{
		cudaFree( cuObj);
	}
	return 0;
}
