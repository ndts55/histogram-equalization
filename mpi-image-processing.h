#pragma once

#include "hist-equ.h"

int process_pgm(int rank, int world_size);

int process_ppm(int rank, int world_size);

int process_as_hsl(int rank, int world_size, PPM_IMG &ppm);

int ppm_to_hsl(PPM_IMG &ppm, int rank, int world_size, HSL_IMG *hsl);

int hsl_to_ppm(HSL_IMG &hsl, int rank, int world_size, PPM_IMG *ppm);

float hue_to_rgb(float v0, float v1, float vh);

int ppm_to_yuv(PPM_IMG &ppm, int rank, int world_size, YUV_IMG *yuv);

int yuv_to_ppm(YUV_IMG &yuv, int rank, int world_size, PPM_IMG *ppm);

int process_as_yuv(int rank, int world_size, PPM_IMG &ppm);
