#pragma once

#include "hist-equ.h"

void process_pgm(PGM_IMG &pgm);

void process_ppm(PPM_IMG &ppm);

void process_as_hsl(PPM_IMG &ppm);

HSL_IMG ppm_to_hsl(PPM_IMG &ppm);

PPM_IMG hsl_to_ppm(HSL_IMG &hsl);

float hue_to_rgb(float v0, float v1, float vh);

YUV_IMG ppm_to_yuv(PPM_IMG &ppm);

PPM_IMG yuv_to_ppm(YUV_IMG &yuv);

void process_as_yuv(PPM_IMG &ppm);
