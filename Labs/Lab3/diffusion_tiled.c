#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>
#include <sys/mman.h>

#define REAL float
#define NX (256)
#define NXP nx

#ifndef M_PI
#define M_PI (3.1415926535897932384626)
#endif


void init(REAL *buff, const int nx, const int ny, const int nz,
      const REAL kx, const REAL ky, const REAL kz,
      const REAL dx, const REAL dy, const REAL dz,
      const REAL kappa, const REAL time) {
   REAL ax, ay, az;
   int jz, jy, jx;
   ax = exp(-kappa*time*(kx*kx));
   ay = exp(-kappa*time*(ky*ky));
   az = exp(-kappa*time*(kz*kz));
   for (jz = 0; jz < nz; jz++) {
      for (jy = 0; jy < ny; jy++) {
         for (jx = 0; jx < nx; jx++) {
            int j = jz*NXP*ny + jy*NXP + jx;
            REAL x = dx*((REAL)(jx + 0.5));
            REAL y = dy*((REAL)(jy + 0.5));
            REAL z = dz*((REAL)(jz + 0.5));
            REAL f0 = (REAL)0.125
               *(1.0 - ax*cos(kx*x))
               *(1.0 - ay*cos(ky*y))
               *(1.0 - az*cos(kz*z));
            buff[j] = f0;
         }
      }
   }
}

REAL accuracy(const REAL *b1, REAL *b2, const int len) {
   REAL err = 0.0;
   int i;
   for (i = 0; i < len; i++) {
      err += (b1[i] - b2[i]) * (b1[i] - b2[i]);
   }
   return (REAL)sqrt(err/len);
}

void diffusion_tiled(REAL *restrict f1, REAL *restrict f2,
      int nx, int ny, int nz,
      REAL ce, REAL cw, REAL cn, REAL cs, REAL ct,
      REAL cb, REAL cc, REAL dt, int count) {

   unsigned long tsc;
   int nthreads;

#pragma omp parallel
   {
      REAL *f1_t = f1;
      REAL *f2_t = f2;
      int mythread;

      for (int i = 0; i < count; ++i) {
#define YBF 16
#pragma omp for collapse(2)
         for (int yy = 0; yy < ny; yy += YBF) {
            for (int z = 0; z < nz; z++) {
               int ymax = yy + YBF;
               if (ymax >= ny) ymax = ny;
               for (int y = yy; y < ymax; y++) {
                  int x;
                  int c, n, s, b, t;
                  x = 0;
                  c =  x + y * NXP + z * NXP * ny;
                  n = (y == 0)    ? c : c - NXP;
                  s = (y == ny-1) ? c : c + NXP;
                  b = (z == 0)    ? c : c - NXP * ny;
                  t = (z == nz-1) ? c : c + NXP * ny;
                  f2_t[c] = cc * f1_t[c] + cw * f1_t[c] + ce * f1_t[c+1]
                     + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
#pragma simd
                  for (x = 1; x < nx-1; x++) {
                     ++c;
                     ++n;
                     ++s;
                     ++b;
                     ++t;
                     f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c+1]
                        + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
                  }
                  ++c;
                  ++n;
                  ++s;
                  ++b;
                  ++t;
                  f2_t[c] = cc * f1_t[c] + cw * f1_t[c-1] + ce * f1_t[c]
                     + cs * f1_t[s] + cn * f1_t[n] + cb * f1_t[b] + ct * f1_t[t];
               }
            }
         }
         REAL *t = f1_t;
         f1_t = f2_t;
         f2_t = t;
      }
   }
   return;
}

static double cur_second(void) {
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}


void dump_result(REAL *f, int nx, int ny, int nz, char *out_path) {
   FILE *out = fopen(out_path, "w");
   assert(out);
   size_t nitems = nx * ny * nz;
   fwrite(f, sizeof(REAL), nitems, out);
   fclose(out);
}

int main(int argc, char *argv[])
{

   struct timeval time_begin, time_end;

   int    nx    = NX;
   int    ny    = NX;
   int    nz    = NX;

   REAL *f1 = (REAL *)malloc(sizeof(REAL)*NX*NX*NX);
   REAL *f2 = (REAL *)malloc(sizeof(REAL)*NX*NX*NX);
   assert(f1 != MAP_FAILED);
   assert(f2 != MAP_FAILED);
   REAL *answer = (REAL *)malloc(sizeof(REAL) * NXP*ny*nz);
   REAL *f_final = NULL;

   REAL   time  = 0.0;
   int    count = 0;
   int    nthreads;

   REAL l, dx, dy, dz, kx, ky, kz, kappa, dt;
   REAL ce, cw, cn, cs, ct, cb, cc;

#pragma omp parallel
#pragma omp master
   nthreads = omp_get_num_threads();

 
   l = 1.0;
   kappa = 0.1;
   dx = dy = dz = l / nx;
   kx = ky = kz = 2.0 * M_PI;
   dt = 0.1*dx*dx / kappa;
   count = 0.1 / dt;

   f_final = (count % 2)? f2 : f1;

   init(f1, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);

   ce = cw = kappa*dt/(dx*dx);
   cn = cs = kappa*dt/(dy*dy);
   ct = cb = kappa*dt/(dz*dz);
   cc = 1.0 - (ce + cw + cn + cs + ct + cb);

   printf("Running diffusion kernel %d times\n", count); fflush(stdout);
   gettimeofday(&time_begin, NULL);
   diffusion_tiled(f1, f2, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc,
         dt, count);
   gettimeofday(&time_end, NULL);
   time = count * dt;
   dump_result(f_final, nx, ny, nz, "diffusion_result.dat");

   init(answer, nx, ny, nz, kx, ky, kz, dx, dy, dz, kappa, time);
   REAL err = accuracy(f_final, answer, nx*ny*nz);
   double elapsed_time = (time_end.tv_sec - time_begin.tv_sec)
      + (time_end.tv_usec - time_begin.tv_usec)*1.0e-6;
   REAL mflops = (nx*ny*nz)*13.0*count/elapsed_time * 1.0e-06;
   double thput = (nx * ny * nz) * sizeof(REAL) * 3.0 * count
      / elapsed_time * 1.0e-09;

   fprintf(stderr, "Elapsed time : %.3f (s)\n", elapsed_time);
   fprintf(stderr, "FLOPS        : %.3f (MFlops)\n", mflops);
   fprintf(stderr, "Throughput   : %.3f (GB/s)\n", thput);
   fprintf(stderr, "Accuracy     : %e\n", err);

   free(f1);
   free(f2);
   return 0;
}