typedef struct {
	float pos[3], v[3], r;
	int isCollision;
} Object;

void myFindSweepDirection( Object *cuObj, int *cuSweepDir, int N);
void mySort( Object *cuObj, int *cuSweepDir, int N);
void mySAP( Object *cuObj, int *cuSweepDir, int N);
void myMoveObject( Object *cuObj, int N, int Boundary, float FrameTime);
