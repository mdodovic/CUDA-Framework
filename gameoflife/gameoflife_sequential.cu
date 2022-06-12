#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define for_x for (int x = 0; x < w; x++)
#define for_y for (int y = 0; y < h; y++)
#define for_xy for_x for_y

void init(unsigned *u, int w, int h) {
    for_xy u[y*w + x] = rand() < RAND_MAX / 10 ? 1 : 0;
}

void evolve(unsigned *univ, int w, int h) {
    // unsigned(*univ)[w] = u;
    unsigned* new_arr = (unsigned *)malloc(h * w * sizeof(unsigned));

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int n = 0;
            for (int y1 = y - 1; y1 <= y + 1; y1++)
                for (int x1 = x - 1; x1 <= x + 1; x1++)
                    if (univ[((y1 + h) % h) * w + ((x1 + w) % w)]) n++;

            if (univ[y*w + x]) n--;
            new_arr[y*w + x] = (n == 3 || (n == 2 && univ[y*w + x]));
        }
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            univ[y*w + x] = new_arr[y*w + x];
        }
    }
}

void game(unsigned *u, int w, int h, int iter) {
    cudaEvent_t start_e = cudaEvent_t();
    cudaEvent_t stop_e = cudaEvent_t();
    cudaEventCreate(&start_e);
    cudaEventCreate(&stop_e);

    for (int i = 0; i < iter; i++) {
        if(i == iter / 2) {
           cudaEventRecord(start_e, 0);
        }

        evolve(u, w, h);

        if(i == iter / 2) {
            float elapsed_e = 0.f;
            cudaEventRecord(stop_e, 0);
            cudaEventSynchronize(stop_e);
            cudaEventElapsedTime(&elapsed_e, start_e, stop_e);
            printf("Evolve time [ms] > %f \n\n", elapsed_e);

        }

    }

	cudaEventDestroy(start_e);
	cudaEventDestroy(stop_e);    
}

int main(int c, char *v[]) {

    cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop = cudaEvent_t();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    int w = 0, h = 0, iter = 0;
    unsigned *u;

    if (c > 1) w = atoi(v[1]);
    if (c > 2) h = atoi(v[2]);
    if (c > 3) iter = atoi(v[3]);
    if (w <= 0) w = 30;
    if (h <= 0) h = 30;
    if (iter <= 0) iter = 1000;

    u = (unsigned *)malloc(w * h * sizeof(unsigned));
    if (!u) exit(1);

    init(u, w, h);

    game(u, w, h, iter);

    free(u);

    float elapsed = 0.f;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Total simulation time [ms] > %f \n\n", elapsed);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
