#include <stdio.h>
typedef struct {
	float pos[3], v[3], r;
	int isCollision;
} Object;

typedef struct {
	float pos[3], r;
	int isCollision;
} FileObject;

void myFindSweepDirection( Object *cuObj, int *cuSweepDir, int N);
void mySort( Object *cuObj, int *cuSweepDir, int N);
void mySAP( Object *cuObj, int *cuSweepDir, int N);
void myPrint( FILE *fptr, Object *cuObj, FileObject *cuFileObj, FileObject *fileObj, int N, int Boundary, float frameTime);
void myMoveObject( Object *cuObj, int N, int Boundary, float FrameTime);
