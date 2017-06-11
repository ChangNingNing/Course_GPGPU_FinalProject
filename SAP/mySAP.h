typedef struct {
	float pos[3], v[3], r;
	int isCollision;
} Object;

void myFindSweepDirection( Object *cuObj, int SweepDir[3], int N);
void mySort( Object *cuObj, int SweepDir[3], int N);
void mySAP( Object *cuObj, int SweepDir[3], int N);
void myMoveObject( Object *cuObj, int N, float FrameTime);
