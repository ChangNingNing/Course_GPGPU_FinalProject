#include "mySAP.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }

/* Cuda Function */
// mySort's kernel
struct myCompare{
	__host__ __device__
	bool operator()(const Object& a, const Object& b){
		return a.pos[0] < b.pos[0];
	}
};

// myMoveObject's kernel
__global__ void CudaMoveObject( Object *cuObj, int N, int Boundary, float FT){
	const float g = -98.0665;	// acceleration of gravity
	const float coef_rest = 0.9;	// coefficient of restitution
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= N) return;

	float tmpPos[3];
	tmpPos[0] = cuObj[id].pos[0] + ( cuObj[id].v[0]*FT );
	tmpPos[1] = cuObj[id].pos[1] + ( cuObj[id].v[1]*FT + g*FT*FT/2 );
	tmpPos[2] = cuObj[id].pos[2] + ( cuObj[id].v[2]*FT );

	if (tmpPos[0] - cuObj[id].r <= 0 || tmpPos[0] + cuObj[id].r >= Boundary){
		cuObj[id].pos[0] = (tmpPos[0]-cuObj[id].r<=0)? cuObj[id].r: Boundary-cuObj[id].r;
		cuObj[id].pos[1] = tmpPos[1];
		cuObj[id].pos[2] = tmpPos[2];

		cuObj[id].v[0] = (cuObj[id].v[0] * -1);
	}
	else if (tmpPos[2] - cuObj[id].r <= 0 || tmpPos[2] + cuObj[id].r >= Boundary){
		cuObj[id].pos[0] = tmpPos[0];
		cuObj[id].pos[1] = tmpPos[1];
		cuObj[id].pos[2] = (tmpPos[2]-cuObj[id].r<=0)? cuObj[id].r: Boundary-cuObj[id].r;

		cuObj[id].v[2] = (cuObj[id].v[2] * -1);
	}
	else if (tmpPos[1] - cuObj[id].r <= 0){
		cuObj[id].pos[0] = tmpPos[0];
		cuObj[id].pos[1] = cuObj[id].r;
		cuObj[id].pos[2] = tmpPos[2];

		cuObj[id].v[1] = (cuObj[id].v[1] * -coef_rest);
	}
	else {
		cuObj[id].pos[0] = tmpPos[0];
		cuObj[id].pos[1] = tmpPos[1];
		cuObj[id].pos[2] = tmpPos[2];

		cuObj[id].v[1] += g*FT;
	}
}
//mySAP kernel
#define sqr(a) (a)*(a)
__global__ void CudaSAP(Object *cuObj, int SweepDir[3], int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= N) return;
    float myPos[3], myR;
    myPos[0] = cuObj[id].pos[0];
    myPos[1] = cuObj[id].pos[1];
    myPos[2] = cuObj[id].pos[2];
    myR = cuObj[id].r;

    int axis;
    if(SweepDir[0])
        axis = 0;
    else if(SweepDir[1])
        axis = 1;
    else
        axis = 2;

    float bound = myPos[axis] + myR;
    for(int i = id + 1; i < N && (cuObj[i].pos[axis] - cuObj[i].r) <= bound ; i++){
        float tmpPos[3], tmpR;
        tmpPos[0] = cuObj[i].pos[0];
        tmpPos[1] = cuObj[i].pos[1];
        tmpPos[2] = cuObj[i].pos[2];
        tmpR = cuObj[i].r;
        float dist = sqr(tmpPos[0]-myPos[0])+sqr(tmpPos[1]-myPos[1])+sqr(tmpPos[2]-myPos[2]);
        if(dist <= (sqr(myR) + sqr(tmpR)) )
            cuObj[i].isCollision = 1;

    }

}

/**/

#define BlockSize 256

void myFindSweepDirection( Object *cuObj, int SweepDir[3], int N){}

void mySort( Object *cuObj, int SweepDir[3], int N){
	thrust::sort(thrust::device, cuObj, cuObj+N, myCompare());
}

void mySAP( Object *cuObj, int SweepDir[3], int N){
	dim3 grid(CeilDiv(N, BlockSize), 1), block(BlockSize, 1);
    CudaSAP<<< grid, block >>>(cuObj, SweepDir, N);
}

void myMoveObject( Object *cuObj, int N, int Boundary, float FT){
	dim3 grid(CeilDiv(N, BlockSize), 1), block(BlockSize, 1);
	CudaMoveObject<<< grid, block >>>(cuObj, N, Boundary, FT);
}
