#include <iostream>
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include "mpi-image-processing.h"

#define VALUE_COUNT 256
#define MAX_VALUE 255
#define MIN_VALUE 0

int main(int argc, char *argv[]) {
    int r;
    int world_size;
    int rank;
    r = MPI_Init(&argc, &argv);
    if (r != 0)return r;
    r = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (r != 0)return r;
    r = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (r != 0)return r;
    r = process_pgm(rank, world_size);
    if (r != 0)return r;
    r = process_ppm(rank, world_size);
    if (r != 0)return r;
    r = MPI_Finalize();
    if (r != 0)return r;
    return 0;
}

template<typename T>
void log(int rank, T t) {
    printf("%d\t%s\n", rank, std::to_string(t).c_str());
}

void construct_chunk_arrays(int *offsets, int *lengths, int total, int n) {
    auto chunk_size = total / n;
    for (auto i = 0; i < n; i++) {
        offsets[i] = i * chunk_size;
        lengths[i] = chunk_size;
    }
    lengths[n - 1] += total % n;
}

int bcast_dims(int *width, int *height) {
    // Broadcast image dimensions.
    int r;
    r = MPI_Bcast(width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (r != 0)return r;
    r = MPI_Bcast(height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (r != 0)return r;
    return 0;
}

int process_array(unsigned char *equalised, int rank, int world_size, int img_size, const unsigned char img[]) {
    int r;
    // Calculate ranges in image data for each process.
    auto img_chunk_offsets = new int[world_size];
    auto img_chunk_lengths = new int[world_size];
    construct_chunk_arrays(img_chunk_offsets, img_chunk_lengths, img_size, world_size);
    auto img_chunk_length = img_chunk_lengths[rank];
//    log(rank, chunk_offset);
    auto partial_img = new unsigned char[img_chunk_length];
    r = MPI_Scatterv(
            &img[0],
            &img_chunk_lengths[0],
            &img_chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            &partial_img[0],
            img_chunk_length,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (r != 0)return r;
    // Calculate partial histogram.
    auto partial_histogram = new int[VALUE_COUNT];
    for (auto i = 0; i < VALUE_COUNT; i++)partial_histogram[i] = 0;
    for (auto i = 0; i < img_chunk_length; i++) {
        partial_histogram[partial_img[i]]++;
    }

    // Allreduce to complete histogram.
    auto histogram = new int[VALUE_COUNT];
    r = MPI_Allreduce(
            &partial_histogram[0],
            &histogram[0],
            VALUE_COUNT,
            MPI_INT,
            MPI_SUM,
            MPI_COMM_WORLD
    );
    if (r != 0)return r;

    // Calculate cumulative.
    auto cumulative = new int[VALUE_COUNT];
    if (rank == 0) {
        cumulative[0] = histogram[0];
        for (auto i = 1; i < VALUE_COUNT; i++)cumulative[i] = cumulative[i - 1] + histogram[i];
    }

    // Find the first non-zero value in complete histogram.
    int min;
    auto i_min = 0;
    do {
        min = histogram[i_min];
        i_min++;
    } while (min == 0 && i_min < VALUE_COUNT);

    // Calculate d.
    auto d = img_size - min;

    // Calculate partial lookup table in scattered range [0, 255].
    auto vc_chunk_offsets = new int[world_size];
    auto vc_chunk_lengths = new int[world_size];
    construct_chunk_arrays(vc_chunk_offsets, vc_chunk_lengths, VALUE_COUNT, world_size);
    auto vc_chunk_length = vc_chunk_lengths[rank];
    auto partial_cumulative = new int[vc_chunk_length];
    r = MPI_Scatterv(
            &cumulative[0],
            &vc_chunk_lengths[0],
            &vc_chunk_offsets[0],
            MPI_INT,
            &partial_cumulative[0],
            vc_chunk_length,
            MPI_INT,
            0,
            MPI_COMM_WORLD
    );
    if (r != 0)return r;
    auto partial_lookup = new int[vc_chunk_length];
    for (auto i = 0; i < vc_chunk_length; i++) {
        auto v = (int) std::round(((double) partial_cumulative[i] - min) * MAX_VALUE / d);
        partial_lookup[i] = std::clamp(v, MIN_VALUE, MAX_VALUE);
    }
    auto lookup = new int[VALUE_COUNT];
    r = MPI_Allgatherv(
            &partial_lookup[0],
            vc_chunk_length,
            MPI_INT,
            &lookup[0],
            &vc_chunk_lengths[0],
            &vc_chunk_offsets[0],
            MPI_INT,
            MPI_COMM_WORLD
    );
    if (r != 0) return r;

    // Equalise the partial image data.
    auto partial_equalised = new unsigned char[img_chunk_length];
    for (auto i = 0; i < img_chunk_length; i++) {
        partial_equalised[i] = lookup[partial_img[i]];
    }

    // Gatherv partial process_array image data into full image.
    r = MPI_Gatherv(
            &partial_equalised[0],
            img_chunk_length,
            MPI_UNSIGNED_CHAR,
            &equalised[0],
            &img_chunk_lengths[0],
            &img_chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (r != 0)return r;
    delete[] partial_img;
    delete[] partial_histogram;
    delete[] partial_equalised;
    delete[] partial_lookup;
    delete[] partial_cumulative;
    delete[] histogram;
    delete[] lookup;
    delete[] cumulative;
    return 0;
}

int process_pgm(int rank, int world_size) {
    int r;
    PGM_IMG pgm;
    if (rank == 0) pgm = read_pgm("in.pgm");

    r = bcast_dims(&pgm.w, &pgm.h);
    if (r != 0)return r;
    auto img_size = pgm.w * pgm.h;
    auto start_time = MPI_Wtime();
    auto equalised = new unsigned char[img_size];
    r = process_array(equalised, rank, world_size, img_size, pgm.img);
    if (r != 0)return r;
    if (rank == 0) {
        pgm.img = equalised;
        auto end_time = MPI_Wtime();
        auto duration = end_time - start_time;
        printf("Took %f to process PGM file.\n", duration);
        write_pgm(pgm, "out.mpi.pgm");
    }
    delete[] equalised;
    return 0;
}

int process_ppm(int rank, int world_size) {
    int r;
    PPM_IMG ppm;
    if (rank == 0)ppm = read_ppm("in.ppm");
    auto start_time = MPI_Wtime();

    // Broadcast image dimensions.
    r = bcast_dims(&ppm.w, &ppm.h);
    if (r != 0)return r;

    r = process_as_hsl(rank, world_size, ppm);
    if (r != 0)return r;
    r = process_as_yuv(rank, world_size, ppm);
    if (r != 0)return r;

    if (rank == 0) {
        auto end_time = MPI_Wtime();
        auto duration = end_time - start_time;
        printf("Took %f to process PPM file.\n", duration);
    }

    return 0;
}

int process_as_hsl(int rank, int world_size, PPM_IMG &ppm) {
    auto start_time = MPI_Wtime();
    int r;
    HSL_IMG hsl;

    r = ppm_to_hsl(ppm, rank, world_size, &hsl);
    if (r != 0)return r;
    auto img_size = hsl.width * hsl.height;
    auto equalised = new unsigned char[img_size];
    r = process_array(equalised, rank, world_size, img_size, hsl.l);
    if (r != 0)return r;
    if (rank == 0) {
        hsl.l = equalised;
    }
    PPM_IMG o_ppm;
    r = hsl_to_ppm(hsl, rank, world_size, &o_ppm);
    if (r != 0) return r;
    if (rank == 0) {
        write_ppm(o_ppm, "out.hsl.mpi.ppm");
        auto end_time = MPI_Wtime();
        auto duration = end_time - start_time;
        printf("Took %f to process PPM file as HSL.\n", duration);
    }
    delete[] equalised;
    return 0;
}

int ppm_to_hsl(PPM_IMG &ppm, int rank, int world_size, HSL_IMG *hsl) {
    int r;
    hsl->width = ppm.w;
    hsl->height = ppm.h;
    auto img_size = hsl->width * hsl->height;
    auto chunk_offsets = new int[world_size];
    auto chunk_lengths = new int[world_size];
    construct_chunk_arrays(chunk_offsets, chunk_lengths, img_size, world_size);
    auto chunk_length = chunk_lengths[rank];
    auto pr = new unsigned char[chunk_length];
    auto pg = new unsigned char[chunk_length];
    auto pb = new unsigned char[chunk_length];
    r = MPI_Scatterv(
            &ppm.img_r[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            &pr[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD);
    if (r != 0)return r;
    r = MPI_Scatterv(
            &ppm.img_g[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            &pg[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD);
    if (r != 0)return r;
    r = MPI_Scatterv(
            &ppm.img_b[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            &pb[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD);
    if (r != 0)return r;
    auto ph = new float[chunk_length];
    auto ps = new float[chunk_length];
    auto pl = new unsigned char[chunk_length];
    for (auto i = 0; i < chunk_length; i++) {
        auto red = float(pr[i]) / MAX_VALUE;
        auto green = float(pg[i]) / MAX_VALUE;
        auto blue = float(pb[i]) / MAX_VALUE;
        auto min = std::min({red, green, blue});
        auto max = std::max({red, green, blue});
        auto delta = max - min;
        auto lightness = (max + min) / 2;
        float hue, saturation;
        if (delta == 0) {
            hue = 0;
            saturation = 0;
        } else {
            saturation = lightness < 0.5 ? delta / (max + min) : delta / (2 - max - min);
            auto delta_r = ((max - red) / 6 + (delta / 2)) / delta;
            auto delta_g = ((max - green) / 6 + (delta / 2)) / delta;
            auto delta_b = ((max - blue) / 6 + (delta / 2)) / delta;
            if (red == max) {
                hue = delta_b - delta_g;
            } else {
                hue = green == max ? 1.0f / 3.0f + delta_r - delta_b : 2.0f / 3.0f + delta_g - delta_r;
            }
        }

        if (hue < 0)hue += 1;
        if (hue > 1)hue -= 1;

        ph[i] = hue;
        ps[i] = saturation;
        pl[i] = (unsigned char) (lightness * 255);
    }
    hsl->h = new float[img_size];
    hsl->s = new float[img_size];
    hsl->l = new unsigned char[img_size];
    r = MPI_Gatherv(
            &ph[0],
            chunk_length,
            MPI_FLOAT,
            &hsl->h[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD);
    if (r != 0) return r;
    r = MPI_Gatherv(
            &ps[0],
            chunk_length,
            MPI_FLOAT,
            &hsl->s[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_FLOAT,
            0,

            MPI_COMM_WORLD);
    if (r != 0) return r;
    r = MPI_Gatherv(
            &pl[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            &hsl->l[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD);
    if (r != 0) return r;
    delete[] ph;
    delete[] ps;
    delete[] pl;
    delete[] pr;
    delete[] pg;
    delete[] pb;
    delete[] chunk_lengths;
    delete[] chunk_offsets;
    return 0;
}

float hue_to_rgb(float v0, float v1, float vh) {
    if (vh < 0)vh += 1;
    if (vh > 1)vh -= 1;
    if ((6 * vh) < 1)return v1 + (v0 - v1) * 6 * vh;
    if ((2 * vh) < 1)return v0;
    if ((3 * vh) < 2)return v1 + (v0 - v1) * (2.0f / 3.0f - vh) * 6;
    return v1;
}

int hsl_to_ppm(HSL_IMG &hsl, int rank, int world_size, PPM_IMG *ppm) {
    int r;
    ppm->w = hsl.width;
    ppm->h = hsl.height;
    auto img_size = ppm->h * ppm->w;
    auto chunk_offsets = new int[world_size];
    auto chunk_lengths = new int[world_size];
    construct_chunk_arrays(chunk_offsets, chunk_lengths, img_size, world_size);
    auto chunk_length = chunk_lengths[rank];
    auto ph = new float[chunk_length];
    auto ps = new float[chunk_length];
    auto pl = new unsigned char[chunk_length];
    r = MPI_Scatterv(
            &hsl.h[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_FLOAT,
            &ph[0],
            chunk_length,
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD
    );
    if (r != 0) return r;
    r = MPI_Scatterv(
            &hsl.s[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_FLOAT,
            &ps[0],
            chunk_length,
            MPI_FLOAT,
            0,
            MPI_COMM_WORLD
    );
    if (r != 0) return r;
    r = MPI_Scatterv(
            &hsl.l[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            &pl[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (r != 0) return r;
    auto pr = new unsigned char[chunk_length];
    auto pg = new unsigned char[chunk_length];
    auto pb = new unsigned char[chunk_length];
    for (auto i = 0; i < chunk_length; i++) {
        auto h = ph[i];
        auto s = ps[i];
        auto l = (float) pl[i] / 255.0f;
        if (s == 0) {
            auto x = (unsigned char) (l * 255);
            pr[i] = x;
            pg[i] = x;
            pb[i] = x;
        } else {
            auto v0 = l < 0.5 ? l * (1 + s) : (l + s) - (s * l);
            auto v1 = 2 * l - v0;
            pr[i] = (unsigned char) (255 * hue_to_rgb(v0, v1, h + 1.0f / 3.0f));
            pg[i] = (unsigned char) (255 * hue_to_rgb(v0, v1, h));
            pb[i] = (unsigned char) (255 * hue_to_rgb(v0, v1, h - 1.0f / 3.0f));
        }
    }
    ppm->img_r = new unsigned char[img_size];
    ppm->img_g = new unsigned char[img_size];
    ppm->img_b = new unsigned char[img_size];
    r = MPI_Gatherv(
            &pr[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            &ppm->img_r[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (r != 0) return r;
    r = MPI_Gatherv(
            &pg[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            &ppm->img_g[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (r != 0) return r;
    r = MPI_Gatherv(
            &pb[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            &ppm->img_b[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (r != 0) return r;
    delete[] ph;
    delete[] ps;
    delete[] pl;
    delete[] pr;
    delete[] pg;
    delete[] pb;
    delete[] chunk_lengths;
    delete[] chunk_offsets;
    return 0;
}

int process_as_yuv(int rank, int world_size, PPM_IMG &ppm) {
    auto start_time = MPI_Wtime();
    int r;
    YUV_IMG yuv;
    r = ppm_to_yuv(ppm, rank, world_size, &yuv);
    if (r != 0)return r;
    auto img_size = yuv.w * yuv.h;
    auto equalised = new unsigned char[img_size];
    r = process_array(equalised, rank, world_size, img_size, yuv.img_y);
    if (r != 0)return r;
    if (rank == 0) {
        yuv.img_y = equalised;
    }
    PPM_IMG o_ppm;
    r = yuv_to_ppm(yuv, rank, world_size, &o_ppm);
    if (r != 0)return r;
    if (rank == 0) {
        write_ppm(o_ppm, "out.yuv.mpi.ppm");
        auto end_time = MPI_Wtime();
        auto duration = end_time - start_time;
        printf("Took %f to process PPM file as YUV.\n", duration);
    }
    delete[] equalised;
    return 0;
}

int ppm_to_yuv(PPM_IMG &ppm, int rank, int world_size, YUV_IMG *yuv) {
    int res;
    yuv->w = ppm.w;
    yuv->h = ppm.h;
    auto img_size = yuv->w * yuv->h;
    auto chunk_lengths = new int[world_size];
    auto chunk_offsets = new int[world_size];
    construct_chunk_arrays(chunk_offsets, chunk_lengths, img_size, world_size);
    auto chunk_length = chunk_lengths[rank];
    auto pr = new unsigned char[chunk_length];
    auto pg = new unsigned char[chunk_length];
    auto pb = new unsigned char[chunk_length];
    res = MPI_Scatterv(
            &ppm.img_r[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            &pr[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (res != 0) return res;
    res = MPI_Scatterv(
            &ppm.img_g[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            &pg[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (res != 0) return res;
    res = MPI_Scatterv(
            &ppm.img_b[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            &pb[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    auto py = new unsigned char[chunk_length];
    auto pu = new unsigned char[chunk_length];
    auto pv = new unsigned char[chunk_length];
    for (auto i = 0; i < chunk_length; i++) {
        auto r = pr[i];
        auto g = pg[i];
        auto b = pb[i];
        py[i] = (unsigned char) (0.299 * r + 0.587 * g + 0.114 * b);
        pu[i] = (unsigned char) (-0.169 * r - 0.331 * g + 0.499 * b + 128);
        pv[i] = (unsigned char) (0.499 * r - 0.418 * g - 0.0813 * b + 128);
    }
    yuv->img_y = new unsigned char[img_size];
    yuv->img_u = new unsigned char[img_size];
    yuv->img_v = new unsigned char[img_size];
    res = MPI_Gatherv(
            &py[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            &yuv->img_y[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (res != 0) return res;
    res = MPI_Gatherv(
            &pu[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            &yuv->img_u[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (res != 0) return res;
    res = MPI_Gatherv(
            &pv[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            &yuv->img_v[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (res != 0) return res;
    delete[] py;
    delete[] pu;
    delete[] pv;
    delete[] pr;
    delete[] pg;
    delete[] pb;
    delete[] chunk_lengths;
    delete[] chunk_offsets;
    return 0;
}

int yuv_to_ppm(YUV_IMG &yuv, int rank, int world_size, PPM_IMG *ppm) {
    int res;
    ppm->h = yuv.h;
    ppm->w = yuv.w;
    auto img_size = ppm->h * ppm->w;
    auto chunk_offsets = new int[world_size];
    auto chunk_lengths = new int[world_size];
    construct_chunk_arrays(chunk_offsets, chunk_lengths, img_size, world_size);
    auto chunk_length = chunk_lengths[rank];
    auto py = new unsigned char[chunk_length];
    auto pu = new unsigned char[chunk_length];
    auto pv = new unsigned char[chunk_length];
    res = MPI_Scatterv(
            &yuv.img_y[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            &py[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (res != 0) return res;
    res = MPI_Scatterv(
            &yuv.img_u[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            &pu[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (res != 0) return res;
    res = MPI_Scatterv(
            &yuv.img_v[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            &pv[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (res != 0) return res;
    auto pr = new unsigned char[chunk_length];
    auto pg = new unsigned char[chunk_length];
    auto pb = new unsigned char[chunk_length];
    for (auto i = 0; i < chunk_length; i++) {
        auto y = (int) py[i];
        auto cb = pu[i] - 128;
        auto cr = pv[i] - 128;
        pr[i] = (unsigned char) std::clamp((int) (y + 1.402 * cr), MIN_VALUE, MAX_VALUE);
        pg[i] = (unsigned char) std::clamp((int) (y - 0.344 * cb - 0.714 * cr), MIN_VALUE, MAX_VALUE);
        pb[i] = (unsigned char) std::clamp((int) (y + 1.772 * cb), MIN_VALUE, MAX_VALUE);
    }
    ppm->img_r = new unsigned char[img_size];
    ppm->img_g = new unsigned char[img_size];
    ppm->img_b = new unsigned char[img_size];
    res = MPI_Gatherv(
            &pr[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            &ppm->img_r[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (res != 0) return res;
    res = MPI_Gatherv(
            &pg[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            &ppm->img_g[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (res != 0) return res;
    res = MPI_Gatherv(
            &pb[0],
            chunk_length,
            MPI_UNSIGNED_CHAR,
            &ppm->img_b[0],
            &chunk_lengths[0],
            &chunk_offsets[0],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
    );
    if (res != 0) return res;
    delete[] py;
    delete[] pu;
    delete[] pv;
    delete[] pr;
    delete[] pg;
    delete[] pb;
    delete[] chunk_lengths;
    delete[] chunk_offsets;
    return 0;
}

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
//    printf("Image size: %d x %d\n", result.w, result.h);

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
