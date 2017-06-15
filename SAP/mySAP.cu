#include "mySAP.h"
#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }

/* Cuda Function */
// myFindSweepDirection's kernel
struct Obj2Diff{
	const int index, num;
	const float mean;
	Obj2Diff(const int index, const int num, const float mean):index(index), num(num),mean(mean) {}
	__host__ __device__
	float operator()(const Object &x) const{
		return ((x.pos[index] - mean)*(x.pos[index] - mean)) / num;
	}
};

struct Obj2Float{
	const int index, num;
	Obj2Float(const int index, const int num): index(index), num(num) {}
	__host__ __device__
	float operator()(const Object &x) const{
		return x.pos[index] / num;
	}
};

struct myFloatAdd{
	__host__ __device__
	float operator()(const float& a, const float& b){
		return a + b;
	}
};

// mySort's kernel
struct myCompare{
	const int index;
	myCompare(const int index): index(index) {}
	__host__ __device__
	bool operator()(const Object& a, const Object& b){
		return (a.pos[index]-a.r) < (b.pos[index]-b.r);
	}
};

// myMoveObject's kernel
__global__ void CudaMoveObject( Object *cuObj, int N, int Boundary, float FT){
	static const float g = -98.0665;	// acceleration of gravity
	static const float coef_rest = 0.9;	// coefficient of restitution
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= N) return;

	float pos[3];
	pos[0] = cuObj[id].pos[0] + ( cuObj[id].v[0]*FT );
	pos[1] = cuObj[id].pos[1] + ( cuObj[id].v[1]*FT + g*FT*FT/2 );
	pos[2] = cuObj[id].pos[2] + ( cuObj[id].v[2]*FT );

	if (pos[0] - cuObj[id].r <= 0 || pos[0] + cuObj[id].r >= Boundary){
		cuObj[id].pos[0] = (pos[0]-cuObj[id].r<=0)? cuObj[id].r: Boundary-cuObj[id].r;
		cuObj[id].pos[1] = pos[1];
		cuObj[id].pos[2] = pos[2];

		cuObj[id].v[0] = (cuObj[id].v[0] * -1);
	}
	else if (pos[2] - cuObj[id].r <= 0 || pos[2] + cuObj[id].r >= Boundary){
		cuObj[id].pos[0] = pos[0];
		cuObj[id].pos[1] = pos[1];
		cuObj[id].pos[2] = (pos[2]-cuObj[id].r<=0)? cuObj[id].r: Boundary-cuObj[id].r;

		cuObj[id].v[2] = (cuObj[id].v[2] * -1);
	}
	else if (pos[1] - cuObj[id].r <= 0){
		cuObj[id].pos[0] = pos[0];
		cuObj[id].pos[1] = cuObj[id].r;
		cuObj[id].pos[2] = pos[2];

		cuObj[id].v[1] = (cuObj[id].v[1] * -coef_rest);
	}
	else {
		cuObj[id].pos[0] = pos[0];
		cuObj[id].pos[1] = pos[1];
		cuObj[id].pos[2] = pos[2];

		cuObj[id].v[1] += g*FT;
	}
}

// mySAP's kernel
#define sqr(a) (a)*(a)
__global__ void CudaSAP(Object *cuObj, int axis, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= N) return;
    float pos[3], _pos[3], r, _r;
	pos[0] = cuObj[id].pos[0];
	pos[1] = cuObj[id].pos[1];
	pos[2] = cuObj[id].pos[2];
	r = cuObj[id].r;

    float bound = pos[axis] + r;
    int num = 0;
    for(int i = id + 1; i < N && (cuObj[i].pos[axis] - cuObj[i].r) <= bound ; i++){
        _pos[0] = cuObj[i].pos[0];
        _pos[1] = cuObj[i].pos[1];
        _pos[2] = cuObj[i].pos[2];
        _r = cuObj[i].r;
        float dist = sqr(_pos[0]-pos[0])+sqr(_pos[1]-pos[1])+sqr(_pos[2]-pos[2]);
        if(dist < (sqr(r + _r)))
			num++;
    }
	cuObj[id].isCollision = num;
}

// mySAP2's kernel
#define nWork 65536
__global__ void CudaWorkload(Object *cuObj, int *cuR, int *cuNT, int axis, int N){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= N) return;

	int right;
	// Binary Search for Right (inclusive)
	{
		int l = id+1, r = N-1, mid;
		float value = cuObj[id].pos[axis] + cuObj[id].r, _v;
		while (l <= r){
			mid = (l + r) / 2;
			_v = cuObj[mid].pos[axis] - cuObj[mid].r;
			if (value > _v) l = mid + 1;
			else r = mid - 1;
		}
		right = r;
	}
	cuR[id] = right;
	cuNT[id] = CeilDiv(right - id, nWork);
	cuObj[id].isCollision = 0;
}

__global__ void CudaSAP_Workload(Object *cuObj, int *cuR, int *cuNT, int axis, int N){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int index;
	// Binary Search for object index
	{
		int l = 0, r = N-1, mid;
		while (l <= r){
			mid = (l + r) / 2;
			if (id >= cuNT[mid]) l = mid + 1;
			else r = mid - 1;
		}
		index = l;
	}
	if (index >= N) return;

	int left, right;
	left = index != 0? id - cuNT[index-1]: id;
	left = left * nWork + index + 1;
	right = left + nWork > cuR[index]? cuR[index]+1: left+nWork;

    float pos[3], _pos[3], r, _r, dist;
    pos[0] = cuObj[index].pos[0];
    pos[1] = cuObj[index].pos[1];
    pos[2] = cuObj[index].pos[2];
    r = cuObj[index].r;

	int num = 0;
	for (int i=left; i<right; i++){
        _pos[0] = cuObj[i].pos[0];
        _pos[1] = cuObj[i].pos[1];
        _pos[2] = cuObj[i].pos[2];
        _r = cuObj[i].r;
        dist = sqr(_pos[0]-pos[0])+sqr(_pos[1]-pos[1])+sqr(_pos[2]-pos[2]);
        if(dist < (sqr(r + _r)))
			num++;
    }
	atomicAdd( &(cuObj[index].isCollision), num);
}


// myPrint's kernel
__global__ void CudaDrawObject(int *cuTmp, Object *cuObj, int N){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= N) return;
	if ( cuObj[id].isDraw) cuTmp[id] = 1;
	else cuTmp[id] = 0;
}

__global__ void CudaPrintObject(FileObject *cuFileObj, Object *cuObj, int *cuTmp, int N){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= N || cuTmp[id] == 0) return;

	if (id == 0 || cuTmp[id] != cuTmp[id-1]){
		int _id = cuTmp[id] - 1;
		cuFileObj[_id].pos[0] = cuObj[id].pos[0];
		cuFileObj[_id].pos[1] = cuObj[id].pos[1];
		cuFileObj[_id].pos[2] = cuObj[id].pos[2];
		cuFileObj[_id].r = cuObj[id].r;
		cuFileObj[_id].isCollision = cuObj[id].isCollision;
	}
}

/**/

#define BlockSize 256

void myFindSweepDirection( Object *cuObj, int *SweepDir, int N){
	float mean, var, maxVar = -1;
	int coor;
	for (int i=0; i<3; i++){
		mean = thrust::transform_reduce(thrust::device, cuObj, cuObj+N,
											Obj2Float(i, N), 0.0f, myFloatAdd());
		var = thrust::transform_reduce(thrust::device, cuObj, cuObj+N,
											Obj2Diff(i, N, mean), 0.0f, myFloatAdd());
		if (var > maxVar){
			maxVar = var;
			coor = i;
		}
	}
	*SweepDir = coor;
}

void mySort( Object *cuObj, int SweepDir, int N){
	thrust::sort(thrust::device, cuObj, cuObj+N, myCompare(SweepDir));
}

void mySAP( Object *cuObj, int SweepDir, int N){
	dim3 grid(CeilDiv(N, BlockSize)), block(BlockSize);
	CudaSAP<<< grid, block >>>(cuObj, SweepDir, N);
}

void mySAP2( Object *cuObj, int *cuR, int *cuNT, int SweepDir, int N){
	dim3 grid(CeilDiv(N, BlockSize)), block(BlockSize);
	CudaWorkload<<< grid, block >>>(cuObj, cuR, cuNT, SweepDir, N);

	thrust::inclusive_scan(thrust::device, cuNT, cuNT+N, cuNT);

	int nThreads;
	cudaMemcpy( &nThreads, cuNT+N-1, sizeof(int), cudaMemcpyDeviceToHost);

	dim3 grid2(CeilDiv(nThreads, 1024)), block2(1024);
	CudaSAP_Workload<<< grid2, block2 >>>(cuObj, cuR, cuNT, SweepDir, N);
}

void myPrint(
	Object *cuObj, FileObject *cuFileObj, FileObject *fileObj, int *cuTmp,
	int N, float frameTime, FILE *fptr
){
	dim3 grid(CeilDiv(N, BlockSize)), block(BlockSize);
	CudaDrawObject<<< grid, block >>>( cuTmp, cuObj, N);
	thrust::inclusive_scan(thrust::device, cuTmp, cuTmp+N, cuTmp);

	CudaPrintObject<<< grid, block >>>(cuFileObj, cuObj, cuTmp, N);

	int nFileObj;
	cudaMemcpy( &nFileObj, cuTmp+N-1, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy( fileObj, cuFileObj, sizeof(FileObject)*nFileObj, cudaMemcpyDeviceToHost);

	fwrite( &nFileObj, sizeof(int), 1, fptr);
	fwrite( fileObj, sizeof(FileObject), nFileObj, fptr);
	fwrite( &frameTime, sizeof(float), 1, fptr);
}

void myMoveObject( Object *cuObj, int N, int Boundary, float FT){
	dim3 grid(CeilDiv(N, BlockSize)), block(BlockSize);
	CudaMoveObject<<< grid, block >>>(cuObj, N, Boundary, FT);
}
