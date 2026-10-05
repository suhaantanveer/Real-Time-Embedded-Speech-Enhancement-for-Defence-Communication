#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "audio_config.h"
#include "features_esp32.h"
#include "gru_inference.h"

static int loadf(const char *path, float *dst, size_t n) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    int ok = fread(dst, sizeof(float), n, f) == n; fclose(f); return ok;
}
static float maxerr(const float *a, const float *b, size_t n) {
    float e=0; for(size_t i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>e)e=d;} return e;
}
int main(void) {
    float mic1[160], mic2[160], py_feat[89], py_gain[22], py_r[129], py_i[129];
    if(!loadf("firmware/test/test_mic1_160.bin", mic1, 160) ||
       !loadf("firmware/test/test_mic2_160.bin", mic2, 160) ||
       !loadf("firmware/test/expected_features_89.bin", py_feat, 89) ||
       !loadf("firmware/test/expected_gains_22.bin", py_gain, 22) ||
       !loadf("firmware/test/expected_stft_real_129.bin", py_r, 129) ||
       !loadf("firmware/test/expected_stft_imag_129.bin", py_i, 129)) {
        fprintf(stderr,"load failed\n"); return 1;
    }
    esp32_features_t feat; features_esp32_init(&feat);
    float c_feat[89], c_r[129], c_i[129];
    features_esp32_extract(&feat, mic1, mic2, true, c_feat, c_r, c_i);
    float e_r=maxerr(c_r,py_r,129), e_i=maxerr(c_i,py_i,129), e_f=maxerr(c_feat,py_feat,89);
    printf("FFT real max %.7g\nFFT imag max %.7g\nFeatures max %.7g\n",e_r,e_i,e_f);
    gru_state_t st; gru_inference_init(&st); float c_gain[22];
    gru_inference_step(&st,c_feat,c_gain);
    float e_g=maxerr(c_gain,py_gain,22); printf("Gains max %.7g\n",e_g);
    bool pass=(e_r<0.02f && e_i<0.02f && e_f<0.06f && e_g<0.01f);
    if(!pass){fprintf(stderr,"FAIL\n"); return 2;}
    printf("PASS\n"); return 0;
}
