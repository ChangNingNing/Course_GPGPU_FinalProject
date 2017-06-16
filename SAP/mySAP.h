#include <stdio.h>
typedef struct {
	float pos[3], v[3], r;
	int isCollision;
	int isDraw;
	int group;
} Object;

typedef struct {
	float pos[3], r;
	int isCollision;
} FileObject;

__host__ __device__ int CeilDiv(int a, int b);
void myFindSweepDirection( Object *cuObj, int *SweepDir, int N);

void mySort( Object *cuObj, int *SweepDir, int N);

void myWS(	Object **cuWS, int *cuNWObj, int nWS, Object *cuObj,
			int *cuTmp, int *SweepDir, int N, int Boundary);

void mySAP( Object *cuObj, int *SweepDir, int N);
void mySAP_WB( Object *cuObj, int *cuR, int *cuNT, int *SweepDir, int N);
void mySAP_WS( 	Object *cuObj, Object **cuWS, int *cuNWObj, int nWS,
				int *cuR, int *cuNT, int *SweepDir, int N);

void myPrint(	Object *cuObj, FileObject *cuFileObj, FileObject *fileObj,
				int *cuTmp, int N, float frameTime, FILE *fptr);

void myMoveObject( Object *cuObj, int N, int Boundary, float FrameTime);
