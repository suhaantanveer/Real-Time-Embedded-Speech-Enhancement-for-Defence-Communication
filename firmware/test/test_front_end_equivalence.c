#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "front_end.h"
static int loadf(const char *p,float*x,int n){FILE*f=fopen(p,"rb");if(!f)return 0;int ok=fread(x,sizeof(float),n,f)==(size_t)n;fclose(f);return ok;}
static float maxe(float*a,float*b,int n){float e=0;for(int i=0;i<n;i++){float d=fabsf(a[i]-b[i]);if(d>e)e=d;}return e;}
int main(void){float in1[160],in2[160],e1[160],e2[160],o1[160],o2[160];if(!loadf("firmware/test/frontend_mic1_in.bin",in1,160)||!loadf("firmware/test/frontend_mic2_in.bin",in2,160)||!loadf("firmware/test/frontend_expected_mic1.bin",e1,160)||!loadf("firmware/test/frontend_expected_mic2.bin",e2,160))return 1;front_end_state_t st;front_end_init(&st);front_end_process_frame(&st,in1,in2,o1,o2,160);float a=maxe(o1,e1,160),b=maxe(o2,e2,160);printf("mic1 %.9g mic2 %.9g\n",a,b);if(a>2e-6f||b>2e-6f)return 2;return 0;}
