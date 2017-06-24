#include <stdio.h>
typedef struct {
	float pos[3], v[3], r;
	int nCollision;
	int isDraw;
	int group;
} Object;

typedef struct {
	float pos[3], r;
	int* nC_ptr;
} ObjectPtr;

typedef struct {
	float pos[3], r;
	int nCollision;
} FileObject;

__host__ __device__ int CeilDiv(int a, int b);
void myFindSweepDirection( Object *cuObj, int *SweepDir, int N);

void mySort( Object *cuObj, int *SweepDir, int N);

void myWS(	ObjectPtr *cuWS, int *cuWSBound, int *cuNWObj, int nWS,
			Object *cuObj, int *cuTmp, int *SweepDir, int N, int Boundary);

void mySAP( Object *cuObj, int *SweepDir, int N);
void mySAP_WB( Object *cuObj, int *cuR, int *cuNT, int *SweepDir, int N);
void mySAP_WS( 	Object *cuObj, ObjectPtr *cuWS, int *cuNWObj, int nWS,
				int *cuR, int *cuNT, int *SweepDir, int N);

void myPrint(	Object *cuObj, FileObject *cuFileObj, FileObject *fileObj,
				int *cuTmp, int N, float frameTime, FILE *fptr);

void myMoveObject( Object *cuObj, int N, int Boundary, float FrameTime);
