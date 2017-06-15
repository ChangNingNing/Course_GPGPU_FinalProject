#include <stdio.h>
typedef struct {
	float pos[3], v[3], r;
	int isCollision;
	int isDraw;
} Object;

typedef struct {
	float pos[3], r;
	int isCollision;
} FileObject;

void myFindSweepDirection( Object *cuObj, int *SweepDir, int N);
void mySort( Object *cuObj, int SweepDir, int N);
void mySAP( Object *cuObj, int SweepDir, int N);
void mySAP2( Object *cuObj, int *cuR, int *cuNT, int SweepDir, int N);
void myPrint( Object *cuObj, FileObject *cuFileObj, FileObject *fileObj, int *cuTmp, int N, float frameTime, FILE *fptr);
void myMoveObject( Object *cuObj, int N, int Boundary, float FrameTime);
