#include <GL/glu.h>
#include <GL/glut.h>
#include <iostream>
#include <stdio.h>
#include <windows.h>

#define MAXN (1 << 20)
typedef struct {
	float pos[3], r;
	int isCollision;
} Object;

Object obj[MAXN];
FILE *fptr;
int RadiusN = 7;
float Radius[] = { 1.5, 3, 6, 12, 24, 48, 96};
float RadiusRGB[][3]={	0.5, 0.0, 0.0,
						0.0, 0.5, 0.0,
						0.0, 0.0, 0.5,
						0.5, 0.5, 0.0,
						0.5, 0.0, 0.5,
						0.0, 0.5, 0.5,
						0.7, 0.0, 0.0};
int RadiusMap[193] = {0};

int Boundary, N, frameNum;

GLdouble eye[3] = { 0, 0, 0};
GLdouble center[3] = { 0, 0, 0};
GLdouble dir[3] = { 0, 1, 0};

void Reshape(int w, int h){
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0f, (GLdouble)h / (GLdouble)w, 1.0, 3*Boundary);
	gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], dir[0], dir[1], dir[2]);
}

void myDrawBoundary(){
	glPushMatrix();
		glColor3f( 0.0, 0.0, 0.0);
		glTranslatef((float)Boundary/2, (float)Boundary/2, (float)Boundary/2);
		glutWireCube(Boundary);
	glPopMatrix();

	glPushMatrix();
		glTranslatef((float)Boundary/2, -1, (float)Boundary/2);
		glScalef( 1.0, (float)1/Boundary, 1.0);
		glutSolidCube(Boundary);
		glScalef( 1.0, (float)Boundary, 1.0);
	glPopMatrix();
}

void myDrawSimulation(){
	for (int i=0; i<frameNum; i++){
		int cIndex = RadiusMap[(int)obj[i].r];
		glPushMatrix();
			/* For correctness
			if (obj[i].isCollision)
				glColor3f( 1, 0, 0);
			else
				glColor3f( 0, 1, 0);
			*/
			glColor3f( RadiusRGB[cIndex][0], RadiusRGB[cIndex][1], RadiusRGB[cIndex][2]);
			glTranslatef( obj[i].pos[0], obj[i].pos[1], obj[i].pos[2]);
			
			/* For correctness
			glutSolidSphere( obj[i].r, 30, 30);
			*/
			if (cIndex < 4)
				glutSolidSphere( obj[i].r, 5, 5);
			else if (cIndex < 6)
				glutSolidSphere( obj[i].r, 10, 10);
			else
				glutSolidSphere( obj[i].r, 20, 20);
		glPopMatrix();
	}
}

void Display( void ){
	glClearColor( 1.0, 1.0, 1.0, 1.0);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluPerspective(60.0f, 1, 1.0, 3*Boundary);
	gluLookAt(eye[0], eye[1], eye[2], center[0], center[1], center[2], dir[0], dir[1], dir[2]);

	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();
	glEnable(GL_DEPTH_TEST);

	myDrawBoundary();
	myDrawSimulation();

	glFlush();
	glutSwapBuffers();
}

void Idle( void ){
	float frameTime = 0;
	fread( &frameNum, sizeof(int), 1, fptr);
	fread( obj, sizeof(Object), frameNum, fptr);
	fread( &frameTime, sizeof(float), 1, fptr);

	printf("%d %f\n", frameNum, (float)1 / frameTime);
	fflush(stdout);

	glutPostRedisplay();
	Sleep((unsigned int)(frameTime * 1000));
}

int main(int argc, char *argv[]){
	// Initial
	{
		if (argc < 2){
			puts("usage: ./main.exe [log file]");
			return 0;
		}

		for (int i=0; i<RadiusN; i++)
			RadiusMap[(int)Radius[i]] = i;
		fptr = fopen(argv[1], "rb");
		if (!fptr){
			printf("open log file error.\n");
			exit(1);
		}
		fread( &Boundary, sizeof(int), 1, fptr);
		fread( &N, sizeof(int), 1, fptr);
		frameNum = N;

		eye[0] = eye[1] = (float)Boundary / 2;
		eye[2] = (float)Boundary * 2;
		center[0] = center[1] = center[2] = (float)Boundary / 2;
	}

	// openGL
	{
		glutInitDisplayMode( GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);

		glutInitWindowPosition(100, 100);
		glutInitWindowSize(600, 600);
		glutCreateWindow("GPGPU CD Simulation");
		glutDisplayFunc(Display);
		glutReshapeFunc(Reshape);
		glutIdleFunc(Idle);
		glutMainLoop();
	}
	return 0;
}
