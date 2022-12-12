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
    std::cout << "Processing PPM Image File." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    process_as_hsl(in_ppm);
    process_as_yuv(in_ppm);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Took " << duration << " to process PPM file." << std::endl;
}

void process_as_hsl(PPM_IMG &ppm) {
    // Take start time for HSL operation.
    // Create HSL_IMG from input image.
    // Calculate histogram on L.
    // Equalise L.
    // Convert HSL_IMG to PPM_IMG.
    // Take end time for HSL operation and display duration.
    // Write resulting PPM_IMG to disk as "out.hsl.ppm".
    std::cout << "Processing PPM Image as HSL." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto hsl = ppm_to_hsl(ppm);
    int histogram[VALUE_COUNT];
    int cumulative[VALUE_COUNT];
    unsigned char lookup[VALUE_COUNT];
    auto size = hsl.height * hsl.width;
    int min = 0;
    int d = 0;
    auto equalised_l = new unsigned char[size];
#pragma omp parallel default(none) shared(hsl, ppm, histogram, cumulative, lookup, size, min, d, equalised_l)
    {
#pragma omp for
        for (auto i = 0; i < VALUE_COUNT; i++) {
            histogram[i] = 0;
            cumulative[i] = 0;
            lookup[i] = 0;
        }
#pragma omp for reduction(+:histogram)
        for (auto i = 0; i < size; i++) {
            histogram[hsl.l[i]]++;
        }
#pragma omp single nowait
        {
            cumulative[0] = histogram[0];
            for (auto i = 1; i < VALUE_COUNT; i++) {
                cumulative[i] = cumulative[i - 1] + histogram[i];
            }
        }
#pragma omp single
        {
            auto i = 0;
            do {
                min = histogram[i];
                i++;
            } while (min == 0);
            d = size - min;
        }
#pragma omp for
        for (auto i = 0; i < VALUE_COUNT; i++) {
            auto v = (int) std::round(((double) cumulative[i] - min) * MAX_VALUE / d);
            lookup[i] = std::clamp(v, MIN_VALUE, MAX_VALUE);
        }
#pragma omp for
        for (auto i = 0; i < size; i++) {
            equalised_l[i] = lookup[hsl.l[i]];
        }
    }
    hsl.l = equalised_l;
    auto result = hsl_to_ppm(hsl);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Took " << duration << " to process PPM as HSL." << std::endl;
    write_ppm(result, "out.hsl.ppm");
    delete[] result.img_r;
    delete[] result.img_g;
    delete[] result.img_b;
    delete[] hsl.h;
    delete[] hsl.s;
    delete[] hsl.l;
}

HSL_IMG ppm_to_hsl(PPM_IMG &ppm) {
    HSL_IMG hsl;
    hsl.width = ppm.w;
    hsl.height = ppm.h;
    auto size = hsl.width * hsl.height;
    hsl.h = new float[size];
    hsl.s = new float[size];
    hsl.l = new unsigned char[size];
    float h, s, l;
#pragma omp parallel for default(none) shared(size, hsl, ppm) private(h, s, l)
    for (auto i = 0; i < size; i++) {
        auto r = float(ppm.img_r[i]) / MAX_VALUE;
        auto g = float(ppm.img_g[i]) / MAX_VALUE;
        auto b = float(ppm.img_b[i]) / MAX_VALUE;
        auto min = std::min(std::min(r, g), b);
        auto max = std::max(std::max(r, g), b);
        auto delta = max - min;
        l = (max + min) / 2;
        if (delta == 0) {
            h = 0;
            s = 0;
        } else {
            s = l < 0.5 ? delta / (max + min) : delta / (2 - max - min);
            auto delta_r = ((max - r) / 6 + (delta / 2)) / delta;
            auto delta_g = ((max - g) / 6 + (delta / 2)) / delta;
            auto delta_b = ((max - b) / 6 + (delta / 2)) / delta;
            if (r == max) {
                h = delta_b - delta_g;
            } else {
                h = g == max ? 1.0f / 3.0f + delta_r - delta_b : 2.0f / 3.0f + delta_g - delta_r;
            }
        }

        if (h < 0)h += 1;
        if (h > 1)h -= 1;

        hsl.h[i] = h;
        hsl.s[i] = s;
        hsl.l[i] = (unsigned char) (l * 255);
    }

    return hsl;
}

PPM_IMG hsl_to_ppm(HSL_IMG &hsl) {
    PPM_IMG ppm;
    ppm.w = hsl.width;
    ppm.h = hsl.height;
    auto size = ppm.w * ppm.h;
    ppm.img_r = new unsigned char[size];
    ppm.img_g = new unsigned char[size];
    ppm.img_b = new unsigned char[size];
    float h, s, l;
#pragma omp parallel for default(none) shared(ppm, hsl, size) private(h, s, l)
    for (auto i = 0; i < size; i++) {
        h = hsl.h[i];
        s = hsl.s[i];
        l = (float) hsl.l[i] / 255.0f;
        if (s == 0) {
            auto x = (unsigned char) (l * 255);
            ppm.img_r[i] = x;
            ppm.img_g[i] = x;
            ppm.img_b[i] = x;
        } else {
            auto v0 = l < 0.5 ? l * (1 + s) : (l + s) - (s * l);
            auto v1 = 2 * l - v0;
            ppm.img_r[i] = (unsigned char) (255 * hue_to_rgb(v0, v1, h + 1.0f / 3.0f));
            ppm.img_g[i] = (unsigned char) (255 * hue_to_rgb(v0, v1, h));
            ppm.img_b[i] = (unsigned char) (255 * hue_to_rgb(v0, v1, h - 1.0f / 3.0f));
        }
    }
    return ppm;
}

float hue_to_rgb(float v0, float v1, float vh) {
    if (vh < 0)vh += 1;
    if (vh > 1)vh -= 1;
    if ((6 * vh) < 1)return v1 + (v0 - v1) * 6 * vh;
    if ((2 * vh) < 1)return v0;
    if ((3 * vh) < 2)return v1 + (v0 - v1) * (2.0f / 3.0f - vh) * 6;
    return v1;
}

void process_as_yuv(PPM_IMG &ppm) {
    // Take start time for YUV operation.
    // Create YUV_IMG from input image.
    // Calculate histogram on Y.
    // Equalise Y.
    // Convert YUV_IMG to PPM_IMG.
    // Take end time for YUV operation and display duration.
    // Write resulting PPM_IMG to disk as "out.yuv.ppm".
    std::cout << "Processing Image File as YUV." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto yuv = ppm_to_yuv(ppm);
    int histogram[VALUE_COUNT];
    int cumulative[VALUE_COUNT];
    unsigned char lookup[VALUE_COUNT];
    auto size = yuv.w * yuv.h;
    int min = 0;
    int d = 0;
    auto equalised_y = new unsigned char[size];
#pragma omp parallel default(none) shared(yuv, ppm, histogram, cumulative, lookup, size, min, d, equalised_y)
    {
#pragma omp for
        for (auto i = 0; i < VALUE_COUNT; i++) {
            histogram[i] = 0;
            cumulative[i] = 0;
            lookup[i] = 0;
        }
#pragma omp for reduction(+:histogram)
        for (auto i = 0; i < size; i++) {
            histogram[yuv.img_y[i]]++;
        }
#pragma omp single nowait
        {
            cumulative[0] = histogram[0];
            for (auto i = 1; i < VALUE_COUNT; i++) {
                cumulative[i] = cumulative[i - 1] + histogram[i];
            }
        }
#pragma omp single
        {
            auto i = 0;
            do {
                min = histogram[i];
                i++;
            } while (min == 0);
            d = size - min;
        }
#pragma omp for
        for (auto i = 0; i < VALUE_COUNT; i++) {
            auto v = (int) std::round(((double) cumulative[i] - min) * MAX_VALUE / d);
            lookup[i] = std::clamp(v, MIN_VALUE, MAX_VALUE);
        }
#pragma omp for
        for (auto i = 0; i < size; i++) {
            equalised_y[i] = lookup[yuv.img_y[i]];
        }
    }
    yuv.img_y = equalised_y;
    auto result = yuv_to_ppm(yuv);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Took " << duration << " to process PPM as YUV." << std::endl;
    write_ppm(result, "out.yuv.ppm");
    delete[] result.img_r;
    delete[] result.img_g;
    delete[] result.img_b;
    delete[] yuv.img_y;
    delete[] yuv.img_u;
    delete[] yuv.img_v;
}

YUV_IMG ppm_to_yuv(PPM_IMG &ppm) {
    YUV_IMG yuv;
    yuv.h = ppm.h;
    yuv.w = ppm.w;
    auto size = yuv.h * yuv.w;
    yuv.img_y = new unsigned char[size];
    yuv.img_u = new unsigned char[size];
    yuv.img_v = new unsigned char[size];
    unsigned char r, g, b;
#pragma omp parallel for default(none) shared(yuv, ppm, size) private(r, g, b)
    for (auto i = 0; i < size; i++) {
        r = ppm.img_r[i];
        g = ppm.img_g[i];
        b = ppm.img_b[i];
        yuv.img_y[i] = (unsigned char) (0.299 * r + 0.587 * g + 0.114 * b);
        yuv.img_u[i] = (unsigned char) (-0.169 * r - 0.331 * g + 0.499 * b + 128);
        yuv.img_v[i] = (unsigned char) (0.499 * r - 0.418 * g - 0.0813 * b + 128);
    }
    return yuv;
}

PPM_IMG yuv_to_ppm(YUV_IMG &yuv) {
    PPM_IMG ppm;
    ppm.w = yuv.w;
    ppm.h = yuv.h;
    auto size = ppm.w * ppm.h;
    ppm.img_r = new unsigned char[size];
    ppm.img_g = new unsigned char[size];
    ppm.img_b = new unsigned char[size];
    int y, cb, cr;
#pragma omp parallel for default(none) shared(yuv, ppm, size) private(y, cb, cr)
    for (auto i = 0; i < size; i++) {
        y = yuv.img_y[i];
        cb = yuv.img_u[i] - 128;
        cr = yuv.img_v[i] - 128;
        ppm.img_r[i] = (unsigned char) std::clamp((int) (y + 1.402 * cr), MIN_VALUE, MAX_VALUE);
        ppm.img_g[i] = (unsigned char) std::clamp((int) (y - 0.344 * cb - 0.714 * cr), MIN_VALUE, MAX_VALUE);
        ppm.img_b[i] = (unsigned char) std::clamp((int) (y + 1.772 * cb), MIN_VALUE, MAX_VALUE);
    }

    return ppm;
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

