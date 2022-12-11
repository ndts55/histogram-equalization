#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "hist-equ.h"
#include "omp-image-processing.h"
#include "omp.h"

#define VALUE_COUNT 256
#define MAX_VALUE 255
#define MIN_VALUE 0

int main() {
    std::cout << "Contrast MPI" << std::endl;

    auto in_pgm = read_pgm("in.pgm");
    process_pgm(in_pgm);
    free_pgm(in_pgm);
    auto in_ppm = read_ppm("in.ppm");
    process_ppm(in_ppm);
    free_ppm(in_ppm);
}

void process_pgm(PGM_IMG &in_pgm) {
    // Take start time.
    // Initialize necessary variables.
    // Calculate histogram array.
    // Calculate cdf array.
    // Calculate min and d.
    // Calculate mapping from input values to equalised values.
    // Map input image to equalised image.
    // Take end time and display duration.
    // Write equalised image to disk.
    std::cout << "Processing PGM Image File." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto img_size = in_pgm.w * in_pgm.h;
    int histogram[VALUE_COUNT];
    int cumulative[VALUE_COUNT];
    unsigned char lookup[VALUE_COUNT];
    auto min = 0;
    auto d = 0.0;
    PGM_IMG out_pgm;
    out_pgm.h = in_pgm.h;
    out_pgm.w = in_pgm.w;
    out_pgm.img = (unsigned char *) malloc(img_size * sizeof(unsigned char));
#pragma omp parallel default(none) shared(histogram, cumulative, lookup, in_pgm, out_pgm, min, d, img_size)
    {
#pragma omp master
        {
            auto n = omp_get_num_threads();
            printf("Working with %d threads!\n", n);
        }
#pragma omp for
        for (auto i = 0; i < VALUE_COUNT; i++) {
            histogram[i] = 0;
            cumulative[i] = 0;
            lookup[i] = 0;
        }
        // Calculate histogram by counting occurrences.
#pragma omp for reduction(+:histogram)
        for (auto i = 0; i < img_size; i++) {
            histogram[in_pgm.img[i]]++;
        }
        // Annoyingly, this part can only be done by a single thread because each iteration depends
        // on the previous one.
#pragma omp single nowait
        {
            cumulative[0] = histogram[0];
            for (auto i = 1; i < VALUE_COUNT; i++) {
                cumulative[i] = cumulative[i - 1] + histogram[i];
            }
        }
        // Construct lookup table first, because that way we don't have to execute this calculation
        // for each value in the input image, whose number is far greater than VALUE_COUNT.
#pragma omp single
        {
            auto i = 0;
            do {
                min = histogram[i];
                i++;
            } while (min == 0);
            d = img_size - min;
        }
#pragma omp for
        for (auto i = 0; i < VALUE_COUNT; i++) {
            auto v = (int) std::round(((double) cumulative[i] - min) * MAX_VALUE / d);
            lookup[i] = std::clamp(v, MIN_VALUE, MAX_VALUE);
        }
        // Finally, map the input image to the equalised output image.
#pragma omp for
        for (auto i = 0; i < img_size; i++) {
            out_pgm.img[i] = lookup[in_pgm.img[i]];
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Took " << duration << " to process PGM file." << std::endl;
    write_pgm(out_pgm, "out.pgm");
    free(out_pgm.img);
}

void process_ppm(PPM_IMG &in_ppm) {
    // Take start time.
    // Initialize necessary variables.
    // HSL:
    //   Take start time for HSL operation.
    //   Create HSL_IMG from input image.
    //   Calculate histogram on L.
    //   Equalise L.
    //   Convert HSL_IMG to PPM_IMG.
    //   Take end time for HSL operation and display duration.
    //   Write resulting PPM_IMG to disk as "out.hsl.ppm".
    // YUV:
    //   Take start time for YUV operation.
    //   Create YUV_IMG from input image.
    //   Calculate histogram on Y.
    //   Equalise Y.
    //   Convert YUV_IMG to PPM_IMG.
    //   Take end time for YUV operation and display duration.
    //   Write resulting PPM_IMG to disk as "out.yuv.ppm".
    // Take end time and display total duration.
}

PPM_IMG read_ppm(const char *path) {
    FILE *in_file;
    char sbuf[VALUE_COUNT];

    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == nullptr) {
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);


    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d", &result.w);
    fscanf(in_file, "%d", &result.h);
    fscanf(in_file, "%d\n", &v_max);
    printf("Image size: %d x %d\n", result.w, result.h);


    result.img_r = (unsigned char *) malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *) malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *) malloc(result.w * result.h * sizeof(unsigned char));
    ibuf = (char *) malloc(3 * result.w * result.h * sizeof(char));


    fread(ibuf, sizeof(unsigned char), 3 * result.w * result.h, in_file);

    for (i = 0; i < result.w * result.h; i++) {
        result.img_r[i] = ibuf[3 * i + 0];
        result.img_g[i] = ibuf[3 * i + 1];
        result.img_b[i] = ibuf[3 * i + 2];
    }

    fclose(in_file);
    free(ibuf);

    return result;
}

void write_ppm(PPM_IMG img, const char *path) {
    FILE *out_file;
    int i;

    char *obuf = (char *) malloc(3 * img.w * img.h * sizeof(char));

    for (i = 0; i < img.w * img.h; i++) {
        obuf[3 * i + 0] = img.img_r[i];
        obuf[3 * i + 1] = img.img_g[i];
        obuf[3 * i + 2] = img.img_b[i];
    }
    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n", img.w, img.h);
    fwrite(obuf, sizeof(unsigned char), 3 * img.w * img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_ppm(PPM_IMG img) {
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

PGM_IMG read_pgm(const char *path) {
    FILE *in_file;
    char sbuf[VALUE_COUNT];


    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == nullptr) {
        printf("Input file not found!\n");
        exit(1);
    }

    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d", &result.w);
    fscanf(in_file, "%d", &result.h);
    fscanf(in_file, "%d\n", &v_max);
    printf("Image size: %d x %d\n", result.w, result.h);

    result.img = (unsigned char *) malloc(result.w * result.h * sizeof(unsigned char));

    fread(result.img, sizeof(unsigned char), result.w * result.h, in_file);
    fclose(in_file);

    return result;
}

void write_pgm(PGM_IMG img, const char *path) {
    FILE *out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n", img.w, img.h);
    fwrite(img.img, sizeof(unsigned char), img.w * img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img) {
    free(img.img);
}

