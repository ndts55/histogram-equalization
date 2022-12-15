#include <iostream>
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include "mpi-image-processing.h"

#define VALUE_COUNT 256
#define MAX_VALUE 255
#define MIN_VALUE 0

int main(int argc, char *argv[]) {
    int world_size;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)printf("Contrast MPI\n");
    process_pgm(rank, world_size);
//    process_ppm(rank, world_size);
    MPI_Finalize();
}

void process_pgm(int rank, int world_size) {
    // Read image data (if rank == 0).
    // Broadcast image dimensions.
    // Calculate ranges in image data for each process.
    // Scatterv image data.
    // Calculate partial histogram.
    // Allreduce to complete histogram.
    // Calculate cumulative (if rank == 0).
    // Find the first non-zero value in complete histogram.
    // Calculate d = img_size - min.
    // Calculate partial lookup table in scattered range [0, 255].
    // If we can use the scattered image data again just equalise that.
    // Gatherv partial partial_equalised image data into partial_equalised image.
    // Write image to file.

    // Read image data (if rank == 0).
    PGM_IMG pgm;
    if (rank == 0) pgm = read_pgm("in.pgm");

    // Broadcast image dimensions.
    MPI_Bcast(&pgm.w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pgm.h, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate ranges in image data for each process.
    auto img_size = pgm.w * pgm.h;
    auto img_chunk_size = img_size / world_size;
    int img_chunk_offsets[world_size];
    int img_chunk_lengths[world_size];
    for (auto i = 0; i < world_size; i++) {
        img_chunk_offsets[i] = i * img_chunk_size;
        img_chunk_lengths[i] = img_chunk_size;
    }
    img_chunk_lengths[world_size - 1] += img_size % world_size;

    // Scatterv image data.
    auto chunk_length = img_chunk_lengths[rank];
    auto chunk_offset = img_chunk_offsets[rank];
    auto partial_img = new unsigned char[chunk_length];
    MPI_Scatterv(
            &pgm.img[0],
            &img_chunk_lengths[0],
            &img_chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            &partial_img[0],
            img_chunk_lengths[rank],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );

    // Calculate partial histogram.
    int partial_histogram[VALUE_COUNT];
    for (int &i: partial_histogram) i = 0;
    for (auto i = 0; i < chunk_length; i++) {
        partial_histogram[partial_img[i]]++;
    }

    // Allreduce to complete histogram.
    int histogram[VALUE_COUNT];
    MPI_Allreduce(&partial_histogram, &histogram, VALUE_COUNT, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Calculate cumulative.
    int cumulative[VALUE_COUNT];
    if (rank == 0) {
        cumulative[0] = histogram[0];
        for (auto i = 1; i < VALUE_COUNT; i++) cumulative[i] = cumulative[i - 1] + histogram[i];
    }

    // Find the first non-zero value in complete histogram.
    int min;
    auto i_min = 0;
    do {
        min = histogram[i_min];
        i_min++;
    } while (min == 0 && i_min < VALUE_COUNT);

    // Calculate d = img_size - min.
    auto d = img_size - min;

    // Calculate partial lookup table in scattered range [0, 255].
    int vc_chunk_offsets[world_size];
    int vc_chunk_lengths[world_size];
    int vc_chunk_size = VALUE_COUNT / world_size;
    for (auto i = 0; i < world_size; i++) {
        vc_chunk_offsets[i] = i * vc_chunk_size;
        vc_chunk_lengths[i] = vc_chunk_size;
    }
    vc_chunk_lengths[world_size - 1] += VALUE_COUNT % world_size;
    int partial_cumulative[vc_chunk_lengths[rank]];
    MPI_Scatterv(
            &cumulative,
            vc_chunk_lengths,
            vc_chunk_offsets,
            MPI_INT,
            &partial_cumulative,
            vc_chunk_lengths[rank],
            MPI_INT,
            0,
            MPI_COMM_WORLD
    );
    int partial_lookup[vc_chunk_lengths[rank]];
    for (auto i = 0; i < vc_chunk_lengths[rank]; i++) {
        auto v = (int) std::round(((double) partial_cumulative[i] - min) * MAX_VALUE / d);
        partial_lookup[i] = std::clamp(v, MIN_VALUE, MAX_VALUE);
    }
    int lookup[VALUE_COUNT];
    MPI_Allgatherv(
            &partial_lookup,
            vc_chunk_lengths[rank],
            MPI_INT,
            &lookup,
            reinterpret_cast<const int *>(VALUE_COUNT),
            vc_chunk_offsets,
            MPI_INT,
            MPI_COMM_WORLD
    );

    // If we can use the scattered image data again just equalise that.
    unsigned char partial_equalised[img_chunk_lengths[rank]];
    for (auto i = 0; i < img_chunk_lengths[rank]; i++) {
        partial_equalised[i] = lookup[partial_img[i]];
    }
    // Gatherv partial partial_equalised image data into partial_equalised image.
    unsigned char equalised[img_size];
    MPI_Gatherv(
            &partial_equalised,
            img_chunk_lengths[rank],
            MPI_UNSIGNED_CHAR,
            &equalised,
            &img_size,
            img_chunk_offsets,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    // Write image to file.
    if (rank == 0) {
        pgm.img = equalised;
        write_pgm(pgm, "out.mpi.pgm");
        free(pgm.img);
    }
}

//void process_ppm(int rank, int world_size) {}
//
//void process_as_hsl(PPM_IMG &ppm) {}
//
//HSL_IMG ppm_to_hsl(PPM_IMG &ppm) {}
//
//PPM_IMG hsl_to_ppm(HSL_IMG &hsl) {}
//
//float hue_to_rgb(float v0, float v1, float vh) {}
//
//YUV_IMG ppm_to_yuv(PPM_IMG &ppm) {}
//
//PPM_IMG yuv_to_ppm(YUV_IMG &yuv) {}
//
//void process_as_yuv(PPM_IMG &ppm) {}

PPM_IMG read_ppm(const char *path) {
    FILE *in_file;
    char sbuf[256];

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
    char sbuf[256];


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
