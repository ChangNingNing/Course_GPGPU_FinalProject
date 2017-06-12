#include "mySAP.h"
#include <thrust/sort.h>
#include <thrust/scan.h>
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
// mySAP's kernel
#define sqr(a) (a)*(a)
__global__ void CudaSAP(Object *cuObj, int *cuSweepDir, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= N) return;
    float myPos[3], myR;
    myPos[0] = cuObj[id].pos[0];
    myPos[1] = cuObj[id].pos[1];
    myPos[2] = cuObj[id].pos[2];
    myR = cuObj[id].r;

    int axis;
    if(cuSweepDir[0])
        axis = 0;
    else if(cuSweepDir[1])
        axis = 1;
    else
        axis = 2;

    float bound = myPos[axis] + myR;
    int flag = 0;
    for(int i = id + 1; i < N && (cuObj[i].pos[axis] - cuObj[i].r) <= bound ; i++){
        float tmpPos[3], tmpR;
        tmpPos[0] = cuObj[i].pos[0];
        tmpPos[1] = cuObj[i].pos[1];
        tmpPos[2] = cuObj[i].pos[2];
        tmpR = cuObj[i].r;
        float dist = sqr(tmpPos[0]-myPos[0])+sqr(tmpPos[1]-myPos[1])+sqr(tmpPos[2]-myPos[2]);
        if(dist <= (sqr(myR) + sqr(tmpR)) ){
            cuObj[i].isCollision = 1;
			flag = 1;
        }
    }
	if(flag)
		cuObj[id].isCollision = 1;
}

// myPrint's kernel
__global__ void CudaZBuffer( int *cuTmp, Object *cuObj, int N, int Boundary){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= Boundary || idy >= Boundary) return;

	int idz = Boundary * 2 - 1;

	int index = -1;
	float minZ = Boundary*2;
	for (int i=0; i<N; i++){
		float dist = cuObj[i].pos[0] - idx;
		if (dist > 32) break;

		dist = sqr(dist) + sqr(cuObj[i].pos[1]-idy);
//		if (dist > sqr(cuObj[i].r)) continue;

		dist = sqrt(sqr(cuObj[i].pos[2]-idz) + dist);
		dist -= cuObj[i].r;
		if (dist < minZ){
			minZ = dist;
			index = i;
		}
	}
	if (index >= 0) cuTmp[index] = 1;
}

__global__ void CudaPruneObject(FileObject *cuFileObj, Object *cuObj, int *cuTmp, int N){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= N) return;

	int index = cuTmp[id];
	if (id != 0 && index == cuTmp[id-1]) return;

	cuFileObj[index].pos[0] = cuObj[id].pos[0];
	cuFileObj[index].pos[1] = cuObj[id].pos[1];
	cuFileObj[index].pos[2] = cuObj[id].pos[2];
	cuFileObj[index].r = cuObj[id].r;
	cuFileObj[index].isCollision = cuObj[id].isCollision;
}

__global__ void CudaPrintObject(FileObject *cuFileObj, Object *cuObj, int N){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= N) return;
	cuFileObj[id].pos[0] = cuObj[id].pos[0];
	cuFileObj[id].pos[1] = cuObj[id].pos[1];
	cuFileObj[id].pos[2] = cuObj[id].pos[2];
	cuFileObj[id].r = cuObj[id].r;
	cuFileObj[id].isCollision = cuObj[id].isCollision;
}

/**/

#define BlockSize 256

void myFindSweepDirection( Object *cuObj, int *SweepDir, int N){}

void mySort( Object *cuObj, int *cuSweepDir, int N){
	thrust::sort(thrust::device, cuObj, cuObj+N, myCompare());
}

void mySAP( Object *cuObj, int *cuSweepDir, int N){
	dim3 grid(CeilDiv(N, BlockSize), 1), block(BlockSize, 1);
	CudaSAP<<< grid, block >>>(cuObj, cuSweepDir, N);
}

void myPrint(
	FILE *fptr, Object *cuObj, FileObject *cuFileObj, FileObject *fileObj,
	int N, int Boundary, float frameTime
){
/*
	int *cuTmp;
	cudaMalloc( &cuTmp, sizeof(int)*N);
	cudaMemset( cuTmp, 0, sizeof(int)*N);

	dim3 grid(CeilDiv(Boundary, 32), CeilDiv(Boundary, 32));
	dim3 block(32, 32);
	// z-buffer concept
	CudaZBuffer<<< grid, block>>>( cuTmp, cuObj, N, Boundary);
	thrust::inclusive_scan(thrust::device, cuTmp, cuTmp+N, cuTmp);
*/
	dim3 grid2(CeilDiv(N, BlockSize), 1), block2(BlockSize, 1);
//	CudaPruneObject<<< grid2, block2 >>>(cuFileObj, cuObj, cuTmp, N);
	CudaPrintObject<<< grid2, block2 >>>(cuFileObj, cuObj, N);

	int nFileObj = N;
//	cudaMemcpy( &nFileObj, cuTmp+N-1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy( fileObj, cuFileObj, sizeof(FileObject)*nFileObj, cudaMemcpyDeviceToHost);

//	printf("%d %lf\n", nFileObj, frameTime);

	fwrite( &nFileObj, sizeof(int), 1, fptr);
	fwrite( fileObj, sizeof(FileObject), nFileObj, fptr);
	fwrite( &frameTime, sizeof(float), 1, fptr);

//	cudaFree(cuTmp);
}

void myMoveObject( Object *cuObj, int N, int Boundary, float FT){
	dim3 grid(CeilDiv(N, BlockSize), 1), block(BlockSize, 1);
	CudaMoveObject<<< grid, block >>>(cuObj, N, Boundary, FT);
}
