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
	Obj2Diff(const int index, const int num, const float mean): index(index), num(num), mean(mean) {}
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
		return (a.pos[index] - a.r) < (b.pos[index] - b.r);
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
	// Set FT = 0 to use this function as reset function.
	cuObj[id].isCollision = 0;
}

// mySplitWorkspace's kernel
struct myCompareByGroup{
	__host__ __device__
	bool operator()(const Object& a, const Object& b){
		return a.group < b.group;
	}
};

__device__ void CudaGroup_add1(int group, int id, int *cuNWObj, Object *cuObj){
	atomicAdd( &(cuNWObj[group]), 1);
	cuObj[id].group = group;
}

__device__ void CudaGroup_add2(	int group, int id, int *cuNWObj, Object *cuObj,
								int nWS, int *cuBound, float *r, float *l){
	atomicAdd( &(cuNWObj[group]), 1);
	cuObj[id].group = group;

	int offset = (group - nWS*nWS)*2;
	atomicMax( &(cuBound[offset+0]), (int)(r[0]-0.001+1));
	atomicMax( &(cuBound[offset+1]), (int)(r[1]-0.001+1));
	offset += ((nWS+1)*(nWS+1)-nWS*nWS)*2;
	atomicMin( &(cuBound[offset+0]), (int)l[0]);
	atomicMin( &(cuBound[offset+1]), (int)l[1]);
}

__global__ void CudaGroup(	Object *cuObj, int *cuBound, int *cuNWObj, int nWS,
							int axis1, int axis2, int N, int Boundary){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= N) return;

	float l[2], r[2];
	l[0] = cuObj[id].pos[axis1] - cuObj[id].r;
	r[0] = cuObj[id].pos[axis1] + cuObj[id].r;
	l[1] = cuObj[id].pos[axis2] - cuObj[id].r;
	r[1] = cuObj[id].pos[axis2] + cuObj[id].r;

	float _r = (float)Boundary / nWS;
	if (r[0] <= _r){ // 0, 1, 6
		if (r[1] <= _r){ // 0
			CudaGroup_add1(0, id, cuNWObj, cuObj);
			return;
		}
		else if (l[1] >= _r){ // 1
			CudaGroup_add1(1, id, cuNWObj, cuObj);
			return;
		}
		else { // 6
			CudaGroup_add2(	6, id, cuNWObj, cuObj,
							nWS, cuBound, r, l);
			return;
		}
	}
	else if (l[0] >= _r){ // 2, 3, 7
		if (r[1] <= _r){ // 2
			CudaGroup_add1(2, id, cuNWObj, cuObj);
			return;
		}
		else if (l[1] >= _r){ // 3
			CudaGroup_add1(3, id, cuNWObj, cuObj);
			return;
		}
		else { // 7
			CudaGroup_add2(	7, id, cuNWObj, cuObj,
							nWS, cuBound, r, l);
			return;
		}
	}
	else { // 4, 5, 8
		if (r[1] <= _r){ // 4
			CudaGroup_add2(	4, id, cuNWObj, cuObj,
							nWS, cuBound, r, l);
			return;
		}
		else if (l[1] >= _r){ // 5
			CudaGroup_add2(	5, id, cuNWObj, cuObj,
							nWS, cuBound, r, l);
			return;
		}
		else { // 8
			CudaGroup_add2(	8, id, cuNWObj, cuObj,
							nWS, cuBound, r, l);
			return;
		}
	}
}

__global__ void CudaSplit(	int *cuTmp, int *cuBound, int *cuNWObj, Object *cuObj,
							int axis1, int axis2, int group, int nWS, int N){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= N) return;

	if (cuObj[id].group == group){
		cuTmp[id] = 1;
		return;
	}

	float r[2], l[2];
	r[0] = cuObj[id].pos[axis1] + cuObj[id].r;
	l[0] = cuObj[id].pos[axis1] - cuObj[id].r;
	r[1] = cuObj[id].pos[axis2] + cuObj[id].r;
	l[1] = cuObj[id].pos[axis2] - cuObj[id].r;

	float _r[2], _l[2];
	int offset = (group - nWS*nWS)*2;
	_r[0] = cuBound[offset + 0];
	_r[1] = cuBound[offset + 1];
	offset += ((nWS+1)*(nWS+1)-nWS*nWS)*2;
	_l[0] = cuBound[offset + 0];
	_l[1] = cuBound[offset + 1];

	if ( (r[0] > _l[0] && r[0] < _r[0]) || (l[0] < _r[0] && l[0] > _l[0])){
		if ( (r[1] > _l[1] && r[1] < _r[1]) || (l[1] < _r[1] && l[1] > _l[1])){
			atomicAdd( &(cuNWObj[group]), 1);
			cuTmp[id] = 1;
			return;
		}
	}
	cuTmp[id] = 0;
}

__global__ void CudaWorkspace( Object **cuWS, int *cuTmp, Object *cuObj, int offset, int N){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= N || cuTmp[id]==0) return;
	if (id != 0 && cuTmp[id] == cuTmp[id-1]) return;

	cuWS[offset + cuTmp[id]-1] = &(cuObj[id]);
}

// mySAP's kernel
#define sqr(a) (a)*(a)
__global__ void CudaSAP( Object *cuObj, int axis, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= N) return;
    float pos[3], _pos[3], r, _r;
	pos[0] = cuObj[id].pos[0];
	pos[1] = cuObj[id].pos[1];
	pos[2] = cuObj[id].pos[2];
	r = cuObj[id].r;

    float bound = pos[axis] + r;
    int num = 0;
    for(int i = id + 1; i < N; i++){
		if (cuObj[i].pos[axis]-cuObj[i].r > bound) break;

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

// mySAP_WB's kernel
#define nWork 65536
__global__ void CudaWorkload( Object *cuWS, int *cuR, int *cuNT, int axis, int N){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= N) return;
	int right;
	// Binary Search for Right (inclusive)
	{
		int l = id+1, r = N-1, mid;
		float value = cuWS[id].pos[axis] + cuWS[id].r, _v;
		while (l <= r){
			mid = (l + r) / 2;
			_v = cuWS[mid].pos[axis] - cuWS[mid].r;
			if (value > _v) l = mid + 1;
			else r = mid - 1;
		}
		right = r;
	}
	cuR[id] = right;
	cuNT[id] = CeilDiv(right - id, nWork);
}

__global__ void CudaSAP_Workload( Object *cuWS, int *cuR, int *cuNT, int axis, int N){
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
    pos[0] = cuWS[index].pos[0];
    pos[1] = cuWS[index].pos[1];
    pos[2] = cuWS[index].pos[2];
    r = cuWS[index].r;

	int num = 0;
	for (int i=left; i<right; i++){
        _pos[0] = cuWS[i].pos[0];
        _pos[1] = cuWS[i].pos[1];
        _pos[2] = cuWS[i].pos[2];
        _r = cuWS[i].r;
        dist = sqr(_pos[0]-pos[0])+sqr(_pos[1]-pos[1])+sqr(_pos[2]-pos[2]);
        if(dist < (sqr(r + _r)))
		{
//			cuWS[i].isCollision = 1;
			num++;
		}
    }
	atomicAdd( &(cuWS[index].isCollision), num);
}

// mySAP_WS's kernel
#define nWorkWS 65536
__global__ void CudaWorkload_WS( Object **cuWS, int *cuR, int *cuNT, int axis, int N){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= N) return;
	int right;
	// Binary Search for Right (inclusive)
	{
		int l = id+1, r = N-1, mid;
		float value = cuWS[id]->pos[axis] + cuWS[id]->r, _v;
		while (l <= r){
			mid = (l + r) / 2;
			_v = cuWS[mid]->pos[axis] - cuWS[mid]->r;
			if (value > _v) l = mid + 1;
			else r = mid - 1;
		}
		right = r;
	}
	cuR[id] = right;
	cuNT[id] = CeilDiv(right - id, nWorkWS);
}

__global__ void CudaSAP_Workload_WS( Object **cuWS, int *cuR, int *cuNT, int axis, int N){
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
	left = left * nWorkWS + index + 1;
	right = left + nWorkWS > cuR[index]? cuR[index]+1: left+nWorkWS;

    float pos[3], _pos[3], r, _r, dist;
    pos[0] = cuWS[index]->pos[0];
    pos[1] = cuWS[index]->pos[1];
    pos[2] = cuWS[index]->pos[2];
    r = cuWS[index]->r;

	int num = 0;
	for (int i=left; i<right; i++){
        _pos[0] = cuWS[i]->pos[0];
        _pos[1] = cuWS[i]->pos[1];
        _pos[2] = cuWS[i]->pos[2];
        _r = cuWS[i]->r;
        dist = sqr(_pos[0]-pos[0])+sqr(_pos[1]-pos[1])+sqr(_pos[2]-pos[2]);
        if(dist < (sqr(r + _r)))
		{
//			cuWS[i]->isCollision = 1;
			num++;
		}
    }
	atomicAdd( &(cuWS[index]->isCollision), num);
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
	float mean, var[3];
	for (int i=0; i<3; i++){
		mean = thrust::transform_reduce(thrust::device, cuObj, cuObj+N,
											Obj2Float(i, N), 0.0f, myFloatAdd());
		var[i] = thrust::transform_reduce(thrust::device, cuObj, cuObj+N,
											Obj2Diff(i, N, mean), 0.0f, myFloatAdd());
	}
	for (int i=0; i<3; i++){
		int max = -1, coor;
		for (int j=0; j<3; j++)
			if (var[j] > max) max = var[j], coor = j;
		var[coor] = -1;
		SweepDir[i] = coor;
	}
}

void mySort( Object *cuObj, int *SweepDir, int N){
	thrust::sort(thrust::device, cuObj, cuObj+N, myCompare(SweepDir[0]));
}

void myWS(	Object **cuWS, int *cuWSBound, int *cuNWObj, int nWS,
			Object *cuObj, int *cuTmp, int *SweepDir, int N, int Boundary)
{
	if (nWS <= 1) return;

	int nGroup = (nWS+1)*(nWS+1);
	int nWSGroup = nGroup - nWS*nWS;

	cudaMemset( cuNWObj, 0, sizeof(int)*nGroup);

	cudaMemset( cuWSBound, 0, sizeof(int)*nWSGroup*2);
	cudaMemset( cuWSBound+nWSGroup*2, 1, sizeof(int)*nWSGroup*2);

	dim3 grid(CeilDiv(N, BlockSize)), block(BlockSize);
	CudaGroup<<< grid, block >>>(	cuObj, cuWSBound, cuNWObj, nWS,
									SweepDir[1], SweepDir[2], N, Boundary);

	for (int i=0, group=i+nWS*nWS; i<nWSGroup; i++, group++){
		CudaSplit<<< grid, block >>>(	cuTmp, cuWSBound, cuNWObj, cuObj,
										SweepDir[1], SweepDir[2], group, nWS, N);
		thrust::inclusive_scan(thrust::device, cuTmp, cuTmp+N, cuTmp);
		CudaWorkspace<<< grid, block >>>( cuWS, cuTmp, cuObj, i*N, N);
	}
}

void mySAP( Object *cuObj, int *SweepDir, int N){
	dim3 grid(CeilDiv(N, BlockSize)), block(BlockSize);
	CudaSAP<<< grid, block >>>( cuObj, SweepDir[0], N);
}

void mySAP_WB( Object *cuObj, int *cuR, int *cuNT, int *SweepDir, int N){
	dim3 grid(CeilDiv(N, BlockSize)), block(BlockSize);
	CudaWorkload<<< grid, block >>>( cuObj, cuR, cuNT, SweepDir[0], N);

	thrust::inclusive_scan(thrust::device, cuNT, cuNT+N, cuNT);

	int nThreads;
	cudaMemcpy( &nThreads, cuNT+N-1, sizeof(int), cudaMemcpyDeviceToHost);
	
	dim3 grid2(CeilDiv(nThreads, 1024)), block2(1024);
	CudaSAP_Workload<<< grid2, block2 >>>( cuObj, cuR, cuNT, SweepDir[0], N);
}

void mySAP_WS(Object *cuObj, Object **cuWS, int *cuNWObj, int nWS, int *cuR, int *cuNT, int *SweepDir, int N){
	if (nWS <= 1){
		mySAP_WB( cuObj, cuR, cuNT, SweepDir, N);
		return;
	}

	int nGroup = (nWS+1)*(nWS+1);
	int nWSGroup = nGroup - nWS*nWS;
	int nWObj[nGroup];
	cudaMemcpy( nWObj, cuNWObj, sizeof(int)*nGroup, cudaMemcpyDeviceToHost);

	for (int i=0, g=i+nWS*nWS; i<nWSGroup; i++, g++){
		Object **_cuWS = cuWS + N*i;
		dim3 grid(CeilDiv(nWObj[g], BlockSize)), block(BlockSize);
		CudaWorkload_WS<<< grid, block >>>( _cuWS, cuR, cuNT, SweepDir[0], nWObj[g]);

		thrust::inclusive_scan(thrust::device, cuNT, cuNT+nWObj[g], cuNT);

		int nThreads;
		cudaMemcpy( &nThreads, cuNT+nWObj[g]-1, sizeof(int), cudaMemcpyDeviceToHost);

		dim3 grid2(CeilDiv(nThreads, 1024)), block2(1024);
		CudaSAP_Workload_WS<<< grid2, block2 >>>( _cuWS, cuR, cuNT, SweepDir[0], nWObj[g]);
	}

	thrust::stable_sort(thrust::device, cuObj, cuObj+N, myCompareByGroup());

	int offset[nWS];
	offset[0] = 0;
	for (int i=1; i<nWS*nWS; i++)
		offset[i] = offset[i-1] + nWObj[i-1];

	for (int i=0; i<nWS*nWS; i++){
		Object *_cuObj = cuObj + offset[i];
		dim3 grid(CeilDiv(nWObj[i], BlockSize)), block(BlockSize);
		CudaWorkload<<< grid, block >>>( _cuObj, cuR, cuNT, SweepDir[0], nWObj[i]);

		thrust::inclusive_scan(thrust::device, cuNT, cuNT+nWObj[i], cuNT);

		int nThreads;
		cudaMemcpy( &nThreads, cuNT+nWObj[i]-1, sizeof(int), cudaMemcpyDeviceToHost);

		dim3 grid2(CeilDiv(nThreads, 1024)), block2(1024);
		CudaSAP_Workload<<< grid2, block2 >>>( _cuObj, cuR, cuNT, SweepDir[0], nWObj[i]);
	}
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
