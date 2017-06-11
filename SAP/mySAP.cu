#include "mySAP.h"

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }

/* Cuda Function */

__global__ void CudaMoveObject( Object *cuObj, int N, float FT){
	const float g = -98.0665;	// acceleration of gravity
	const float coef_rest = 0.9;	// coefficient of restitution
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= N) return;

	float tmpPos[3];
	tmpPos[0] = cuObj[id].pos[0] + ( cuObj[id].v[0]*FT );
	tmpPos[1] = cuObj[id].pos[1] + ( cuObj[id].v[1]*FT + g*FT*FT/2 );
	tmpPos[2] = cuObj[id].pos[2] + ( cuObj[id].v[2]*FT );

	if (tmpPos[1] - cuObj[id].r > 0){
		cuObj[id].pos[0] = tmpPos[0];
		cuObj[id].pos[1] = tmpPos[1];
		cuObj[id].pos[2] = tmpPos[2];

		cuObj[id].v[1] += g*FT;
	}
	else {
		cuObj[id].pos[0] = tmpPos[0];
		cuObj[id].pos[1] = cuObj[id].r;
		cuObj[id].pos[2] = tmpPos[2];

		cuObj[id].v[1] = (cuObj[id].v[1] * -coef_rest);
	}
}

/**/

#define BlockSize 256

void myFindSweepDirection( Object *cuObj, int SweepDir[3], int N){}
void mySort( Object *cuObj, int SweepDir[3], int N){}
void mySAP( Object *cuObj, int SweepDir[3], int N){}

void myMoveObject( Object *cuObj, int N, float FT){
	dim3 grid(CeilDiv(N, BlockSize), 1), block(BlockSize, 1);
	CudaMoveObject<<< grid, block >>>(cuObj, N, FT);
}