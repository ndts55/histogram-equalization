#pragma once

#include "hist-equ.h"

int process_pgm(int rank, int world_size);

int process_ppm(int rank, int world_size);

int process_as_hsl(PPM_IMG &ppm);

HSL_IMG ppm_to_hsl(PPM_IMG &ppm);

PPM_IMG hsl_to_ppm(HSL_IMG &hsl);

float hue_to_rgb(float v0, float v1, float vh);

YUV_IMG ppm_to_yuv(PPM_IMG &ppm);

PPM_IMG yuv_to_ppm(YUV_IMG &yuv);

int process_as_yuv(PPM_IMG &ppm);
