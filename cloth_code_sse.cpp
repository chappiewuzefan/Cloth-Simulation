// #include "./cloth_code.h"
// #include <math.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <emmintrin.h>
// #include <immintrin.h>

// void initMatrix(int n, double UNUSED(mass), double UNUSED(fcon),
//                 int UNUSED(delta), double UNUSED(grav), double sep,
//                 double rball, double offset, double UNUSED(dt), double **x,
//                 double **y, double **z, double **cpx, double **cpy,
//                 double **cpz, double **fx, double **fy, double **fz,
//                 double **vx, double **vy, double **vz, double **oldfx,
//                 double **oldfy, double **oldfz)
// {
//     int nx, ny;

//     // Free any existing
//     free(*x);
//     free(*y);
//     free(*z);
//     free(*cpx);
//     free(*cpy);
//     free(*cpz);

//     // allocate arrays to hold locations of nodes
//     *x = (double *)malloc(n * n * sizeof(double));
//     *y = (double *)malloc(n * n * sizeof(double));
//     *z = (double *)malloc(n * n * sizeof(double));
//     // This is for opengl stuff
//     *cpx = (double *)malloc(n * n * sizeof(double));
//     *cpy = (double *)malloc(n * n * sizeof(double));
//     *cpz = (double *)malloc(n * n * sizeof(double));

//     // initialize coordinates of cloth
//     for (nx = 0; nx < n; nx++)
//     {
//         for (ny = 0; ny < n; ny++)
//         {
//             (*x)[n * nx + ny] = nx * sep - (n - 1) * sep * 0.5 + offset;
//             (*z)[n * nx + ny] = rball + 1;
//             (*y)[n * nx + ny] = ny * sep - (n - 1) * sep * 0.5 + offset;
//             (*cpx)[n * nx + ny] = 0;
//             (*cpz)[n * nx + ny] = 1;
//             (*cpy)[n * nx + ny] = 0;
//         }
//     }

//     // Throw away existing arrays
//     free(*fx);
//     free(*fy);
//     free(*fz);
//     free(*vx);
//     free(*vy);
//     free(*vz);
//     free(*oldfx);
//     free(*oldfy);
//     free(*oldfz);
//     // Alloc new
//     // use calloc instead of malloc (calloc will intialize all the blocks into 0)
//     *fx = (double *)calloc(n * n, sizeof(double));
//     *fy = (double *)calloc(n * n, sizeof(double));
//     *fz = (double *)calloc(n * n, sizeof(double));
//     *vx = (double *)calloc(n * n, sizeof(double));
//     *vy = (double *)calloc(n * n, sizeof(double));
//     *vz = (double *)calloc(n * n, sizeof(double));
//     *oldfx = (double *)malloc(n * n * sizeof(double));
//     *oldfy = (double *)malloc(n * n * sizeof(double));
//     *oldfz = (double *)malloc(n * n * sizeof(double));
// }

// void loopcode(int n, double mass, double fcon, int delta, double grav,
//               double sep, double rball, double xball, double yball,
//               double zball, double dt, double *x, double *y, double *z,
//               double *fx, double *fy, double *fz, double *vx, double *vy,
//               double *vz, double *oldfx, double *oldfy, double *oldfz,
//               double *pe, double *ke, double *te)
// {
//     int i, j;
//     double xdiff, ydiff, zdiff, vmag, damp, rball_div_vmag, cons_5;
//     __m256d cons_3 = _mm256_set_pd(dt, dt, dt, dt);
//     __m256d cons_4 = _mm256_mul_pd(cons_3, _mm256_div_pd(_mm256_set_pd(0.5, 0.5, 0.5, 0.5), _mm256_set_pd(mass, mass, mass, mass)));
//     for (j = 0; j < n; j++)
//     {
//         for (i = 0; i < (n / 4) * 4; i += 4)
//         {
//             // load two double precision
//             __m256d sse_x = _mm256_load_pd(&x[j * n + i]);
//             __m256d sse_y = _mm256_load_pd(&y[j * n + i]);
//             __m256d sse_z = _mm256_load_pd(&z[j * n + i]);
//             __m256d sse_vx = _mm256_load_pd(&vx[j * n + i]);
//             __m256d sse_vy = _mm256_load_pd(&vy[j * n + i]);
//             __m256d sse_vz = _mm256_load_pd(&vz[j * n + i]);
//             __m256d sse_fx = _mm256_load_pd(&fx[j * n + i]);
//             __m256d sse_fy = _mm256_load_pd(&fy[j * n + i]);
//             __m256d sse_fz = _mm256_load_pd(&fz[j * n + i]);

//             _mm256_store_pd(&oldfx[j * n + i], sse_fx);
//             _mm256_store_pd(&oldfy[j * n + i], sse_fy);
//             _mm256_store_pd(&oldfz[j * n + i], sse_fz);

//             _mm256_store_pd(&x[j * n + i], _mm256_add_pd(sse_x, _mm256_mul_pd(cons_3, _mm256_add_pd(sse_vx, _mm256_mul_pd(sse_fx, cons_4)))));
//             _mm256_store_pd(&y[j * n + i], _mm256_add_pd(sse_y, _mm256_mul_pd(cons_3, _mm256_add_pd(sse_vy, _mm256_mul_pd(sse_fy, cons_4)))));
//             _mm256_store_pd(&z[j * n + i], _mm256_add_pd(sse_z, _mm256_mul_pd(cons_3, _mm256_add_pd(sse_vz, _mm256_mul_pd(sse_fz, cons_4)))));
//         }
//         for (i = (n / 4) * 4; i < n; i++)
//         {
//             x[j * n + i] += dt * (vx[j * n + i] + dt * fx[j * n + i] * 0.5 / mass);
//             oldfx[j * n + i] = fx[j * n + i];
//             y[j * n + i] += dt * (vy[j * n + i] + dt * fy[j * n + i] * 0.5 / mass);
//             oldfy[j * n + i] = fy[j * n + i];
//             z[j * n + i] += dt * (vz[j * n + i] + dt * fz[j * n + i] * 0.5 / mass);
//             oldfz[j * n + i] = fz[j * n + i];
//         }
//     }

//     for (j = 0; j < n; j++)
//     {
//         for (i = 0; i < n; i++)
//         {
//             //	apply constraints - push cloth outside of ball
//             xdiff = x[j * n + i] - xball;
//             ydiff = y[j * n + i] - yball;
//             zdiff = z[j * n + i] - zball;
//             vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
//             if (vmag < rball)
//             {
//                 rball_div_vmag = rball / vmag;
//                 x[j * n + i] = xball + xdiff * rball_div_vmag;
//                 y[j * n + i] = yball + ydiff * rball_div_vmag;
//                 z[j * n + i] = zball + zdiff * rball_div_vmag;

//                 cons_5 = (vx[j * n + i] * xdiff + vy[j * n + i] * ydiff + vz[j * n + i] * zdiff) / (vmag * vmag);

//                 vx[j * n + i] = 0.1 * (vx[j * n + i] - xdiff * cons_5);
//                 vy[j * n + i] = 0.1 * (vy[j * n + i] - ydiff * cons_5);
//                 vz[j * n + i] = 0.1 * (vz[j * n + i] - zdiff * cons_5);
//             }
//         }
//     }

//     *pe = eval_pef(n, delta, mass, grav, sep, fcon, x, y, z, fx, fy, fz);

//     // Add a damping factor to eventually set velocity to zero
//     damp = 0.995;
//     __m256d cons_6 = _mm256_set_pd(damp, damp, damp, damp);
//     for (j = 0; j < n; j++)
//     {
//         for (i = 0; i < (n / 4) * 4; i += 4)
//         {
//             __m256d sse_fx = _mm256_load_pd(&fx[j * n + i]);
//             __m256d sse_fy = _mm256_load_pd(&fy[j * n + i]);
//             __m256d sse_fz = _mm256_load_pd(&fz[j * n + i]);
//             __m256d sse_vx = _mm256_load_pd(&vx[j * n + i]);
//             __m256d sse_vy = _mm256_load_pd(&vy[j * n + i]);
//             __m256d sse_vz = _mm256_load_pd(&vz[j * n + i]);
//             __m256d sse_oldfx = _mm256_load_pd(&oldfx[j * n + i]);
//             __m256d sse_oldfy = _mm256_load_pd(&oldfy[j * n + i]);
//             __m256d sse_oldfz = _mm256_load_pd(&oldfz[j * n + i]);

//             _mm256_store_pd(&vx[j * n + i], _mm256_mul_pd(cons_6, _mm256_add_pd(sse_vx, _mm256_mul_pd(cons_4, _mm256_add_pd(sse_fx, sse_oldfx)))));
//             _mm256_store_pd(&vy[j * n + i], _mm256_mul_pd(cons_6, _mm256_add_pd(sse_vy, _mm256_mul_pd(cons_4, _mm256_add_pd(sse_fy, sse_oldfy)))));
//             _mm256_store_pd(&vz[j * n + i], _mm256_mul_pd(cons_6, _mm256_add_pd(sse_vz, _mm256_mul_pd(cons_4, _mm256_add_pd(sse_fz, sse_oldfz)))));
//         }
//         for (i = (n / 4) * 4; i < n; i++)
//         {
//             vx[j * n + i] = (vx[j * n + i] + (fx[j * n + i] + oldfx[j * n + i]) * dt * 0.5 / mass) * damp;
//             vy[j * n + i] = (vy[j * n + i] + (fy[j * n + i] + oldfy[j * n + i]) * dt * 0.5 / mass) * damp;
//             vz[j * n + i] = (vz[j * n + i] + (fz[j * n + i] + oldfz[j * n + i]) * dt * 0.5 / mass) * damp;
//         }
//     }
//     *ke = 0.0;
//     for (j = 0; j < n; j++)
//     {
//         for (i = 0; i < n; i++)
//         {

//             *ke += vx[j * n + i] * vx[j * n + i] + vy[j * n + i] * vy[j * n + i] + vz[j * n + i] * vz[j * n + i];
//         }
//     }
//     *ke = *ke / 2.0;
//     *te = *pe + *ke;
// }

// double eval_pef(int n, int delta, double mass, double grav, double sep,
//                 double fcon, double *x, double *y, double *z, double *fx,
//                 double *fy, double *fz)
// {
//     double pe, rlen, xdiff, ydiff, zdiff, vmag;
//     int nx, ny, dx, dy;

//     pe = 0.0;
//     __m256d sse_sep = _mm256_set_pd(sep, sep, sep, sep);
//     __m256d sse_fcon = _mm256_set_pd(fcon, fcon, fcon, fcon);
//     // loop over particles
//     for (nx = 0; nx < n; nx++)
//     {
//         for (ny = 0; ny < n; ny++)
//         {
//             fx[nx * n + ny] = 0.0;
//             fy[nx * n + ny] = 0.0;
//             fz[nx * n + ny] = -mass * grav;
//             // loop over displacements
//             __m256d sse_nx = _mm256_set_pd(x[nx * n + ny], x[nx * n + ny], x[nx * n + ny], x[nx * n + ny]);
//             __m256d sse_ny = _mm256_set_pd(y[nx * n + ny], y[nx * n + ny], y[nx * n + ny], y[nx * n + ny]);
//             __m256d sse_nz = _mm256_set_pd(z[nx * n + ny], z[nx * n + ny], z[nx * n + ny], z[nx * n + ny]);
//             double sse_pe_list[4] = {0, 0, 0, 0};
//             double sse_fx_list[4] = {0, 0, 0, 0};
//             double sse_fy_list[4] = {0, 0, 0, 0};
//             double sse_fz_list[4] = {0, 0, 0, 0};

//             // dx = nx; MAX(ny - delta, 0 < dy < MIN(ny, n);
//             for (dy = MAX(ny - delta, 0); dy < ny - 4 ; dy += 4)
//             {
//                 // compute reference distance
//                 double sse_dy_rlen_list[4] = {(double)((ny - dy)), (double)((ny - (dy + 1))), (double)((ny - (dy + 2))), (double)((ny - (dy + 3)))};
//                 __m256d sse_dy_rlen = _mm256_load_pd(sse_dy_rlen_list);
//                 __m256d sse_rlen = _mm256_mul_pd(sse_dy_rlen, sse_sep);
//                 // compute actual distance
//                 __m256d sse_dx = _mm256_load_pd(&x[nx * n + dy]);
//                 __m256d sse_dy = _mm256_load_pd(&y[nx * n + dy]);
//                 __m256d sse_dz = _mm256_load_pd(&z[nx * n + dy]);
//                 __m256d sse_xdiff = _mm256_sub_pd(sse_dx, sse_nx);
//                 __m256d sse_ydiff = _mm256_sub_pd(sse_dy, sse_ny);
//                 __m256d sse_zdiff = _mm256_sub_pd(sse_dz, sse_nz);
//                 __m256d sse_vmag = _mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(sse_xdiff, sse_xdiff), _mm256_mul_pd(sse_ydiff, sse_ydiff)), _mm256_mul_pd(sse_zdiff, sse_zdiff)));
//                 // potential energy and force
//                 __m256d sse_vmag_sub_rlen = _mm256_sub_pd(sse_vmag, sse_rlen);
//                 __m256d sse_vmag_sub_rlen_div_vmag = _mm256_div_pd(sse_vmag_sub_rlen, sse_vmag);
//                 _mm256_store_pd(sse_pe_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_vmag_sub_rlen), sse_vmag_sub_rlen));
//                 pe += sse_pe_list[0] + sse_pe_list[1] + sse_pe_list[2] + sse_pe_list[3];
//                 _mm256_store_pd(sse_fx_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_xdiff), sse_vmag_sub_rlen_div_vmag));
//                 fx[nx * n + ny] += sse_fx_list[0] + sse_fx_list[1] + sse_fx_list[2] + sse_fx_list[3];
//                 _mm256_store_pd(sse_fy_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_ydiff), sse_vmag_sub_rlen_div_vmag));
//                 fy[nx * n + ny] += sse_fy_list[0] + sse_fy_list[1] + sse_fy_list[2] + sse_fy_list[3];
//                 _mm256_store_pd(sse_fz_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_zdiff), sse_vmag_sub_rlen_div_vmag));
//                 fz[nx * n + ny] += sse_fz_list[0] + sse_fz_list[1] + sse_fz_list[2] + sse_fz_list[3];
//             }
//             for (dy; dy < ny; dy++)
//             {
//                 // compute reference distance
//                 rlen = (double)((ny - dy)) * sep;
//                 // compute actual distance
//                 xdiff = x[nx * n + dy] - x[nx * n + ny];
//                 ydiff = y[nx * n + dy] - y[nx * n + ny];
//                 zdiff = z[nx * n + dy] - z[nx * n + ny];
//                 vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
//                 // potential energy and force
//                 pe += fcon * (vmag - rlen) * (vmag - rlen);
//                 fx[nx * n + ny] += fcon * xdiff * (vmag - rlen) / vmag;
//                 fy[nx * n + ny] += fcon * ydiff * (vmag - rlen) / vmag;
//                 fz[nx * n + ny] += fcon * zdiff * (vmag - rlen) / vmag;
//             }

//             // dx = nx; ny < dy < MIN(ny + delta + 1, n)
//             for (dy = ny + 1; dy < MIN(ny + delta + 1, n) - 4; dy += 4)
//             {
//                 // compute reference distance
//                 double sse_dy_rlen_list[4] = {(double)((ny - dy) * (ny - dy)), (double)((ny - (dy + 1)) * (ny - (dy + 1))), (double)((ny - (dy + 2)) * (ny - (dy + 2))), (double)((ny - (dy + 3)) * (ny - (dy + 3)))};
//                 __m256d sse_dy_rlen = _mm256_load_pd(sse_dy_rlen_list);
//                 __m256d sse_rlen = _mm256_mul_pd(_mm256_sqrt_pd(sse_dy_rlen), sse_sep);
//                 // compute actual distance
//                 __m256d sse_dx = _mm256_load_pd(&x[nx * n + dy]);
//                 __m256d sse_dy = _mm256_load_pd(&y[nx * n + dy]);
//                 __m256d sse_dz = _mm256_load_pd(&z[nx * n + dy]);
//                 __m256d sse_xdiff = _mm256_sub_pd(sse_dx, sse_nx);
//                 __m256d sse_ydiff = _mm256_sub_pd(sse_dy, sse_ny);
//                 __m256d sse_zdiff = _mm256_sub_pd(sse_dz, sse_nz);
//                 __m256d sse_vmag = _mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(sse_xdiff, sse_xdiff), _mm256_mul_pd(sse_ydiff, sse_ydiff)), _mm256_mul_pd(sse_zdiff, sse_zdiff)));
//                 // potential energy and force
//                 __m256d sse_vmag_sub_rlen = _mm256_sub_pd(sse_vmag, sse_rlen);
//                 __m256d sse_vmag_sub_rlen_div_vmag = _mm256_div_pd(sse_vmag_sub_rlen, sse_vmag);
//                 _mm256_store_pd(sse_pe_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_vmag_sub_rlen), sse_vmag_sub_rlen));
//                 pe += sse_pe_list[0] + sse_pe_list[1] + sse_pe_list[2] + sse_pe_list[3];
//                 _mm256_store_pd(sse_fx_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_xdiff), sse_vmag_sub_rlen_div_vmag));
//                 fx[nx * n + ny] += sse_fx_list[0] + sse_fx_list[1] + sse_fx_list[2] + sse_fx_list[3];
//                 _mm256_store_pd(sse_fy_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_ydiff), sse_vmag_sub_rlen_div_vmag));
//                 fy[nx * n + ny] += sse_fy_list[0] + sse_fy_list[1] + sse_fy_list[2] + sse_fy_list[3];
//                 _mm256_store_pd(sse_fz_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_zdiff), sse_vmag_sub_rlen_div_vmag));
//                 fz[nx * n + ny] += sse_fz_list[0] + sse_fz_list[1] + sse_fz_list[2] + sse_fz_list[3];
//             }
//             for (dy; dy < MIN(ny + delta + 1, n); dy++)
//             {
//                 // compute reference distance
//                 rlen = sqrt((double)((ny - dy) * (ny - dy))) * sep;
//                 // compute actual distance
//                 xdiff = x[nx * n + dy] - x[nx * n + ny];
//                 ydiff = y[nx * n + dy] - y[nx * n + ny];
//                 zdiff = z[nx * n + dy] - z[nx * n + ny];
//                 vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
//                 // potential energy and force
//                 pe += fcon * (vmag - rlen) * (vmag - rlen);
//                 fx[nx * n + ny] += fcon * xdiff * (vmag - rlen) / vmag;
//                 fy[nx * n + ny] += fcon * ydiff * (vmag - rlen) / vmag;
//                 fz[nx * n + ny] += fcon * zdiff * (vmag - rlen) / vmag;
//             }

//             // MAX(nx - delta, 0) <= dx < nx;
//             for (dx = MAX(nx - delta, 0); dx < nx; dx++)
//             {
//                 double sse_dx_rlen_list[4] = {(double)((nx - dx) * (nx - dx)), (double)((nx - dx) * (nx - dx)), (double)((nx - dx) * (nx - dx)), (double)((nx - dx) * (nx - dx))};
//                 __m256d sse_dx_rlen = _mm256_load_pd(sse_dx_rlen_list);

//                 // MAX(ny - delta, 0) < dy < MIN(ny + delta + 1, n)
//                 for (dy = MAX(ny - delta, 0); dy < MIN(ny + delta + 1, n) - 4; dy+=4)
//                 {
//                     // compute reference distance
//                     double sse_dy_rlen_list[4] = {(double)((ny - dy) * (ny - dy)), (double)((ny - (dy + 1)) * (ny - (dy + 1))), (double)((ny - (dy + 2)) * (ny - (dy + 2))), (double)((ny - (dy + 3)) * (ny - (dy + 3)))};
//                     __m256d sse_dy_rlen = _mm256_load_pd(sse_dy_rlen_list);
//                     __m256d sse_rlen = _mm256_mul_pd(_mm256_sqrt_pd(_mm256_add_pd(sse_dx_rlen, sse_dy_rlen)), sse_sep);

//                     // compute actual distance
//                     __m256d sse_dx = _mm256_load_pd(&x[dx * n + dy]);
//                     __m256d sse_dy = _mm256_load_pd(&y[dx * n + dy]);
//                     __m256d sse_dz = _mm256_load_pd(&z[dx * n + dy]);
//                     __m256d sse_xdiff = _mm256_sub_pd(sse_dx, sse_nx);
//                     __m256d sse_ydiff = _mm256_sub_pd(sse_dy, sse_ny);
//                     __m256d sse_zdiff = _mm256_sub_pd(sse_dz, sse_nz);
//                     __m256d sse_vmag = _mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(sse_xdiff, sse_xdiff), _mm256_mul_pd(sse_ydiff, sse_ydiff)), _mm256_mul_pd(sse_zdiff, sse_zdiff)));

//                     // potential energy and force
//                     __m256d sse_vmag_sub_rlen = _mm256_sub_pd(sse_vmag, sse_rlen);
//                     __m256d sse_vmag_sub_rlen_div_vmag = _mm256_div_pd(sse_vmag_sub_rlen, sse_vmag);
//                     _mm256_store_pd(sse_pe_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_vmag_sub_rlen), sse_vmag_sub_rlen));
//                     pe += sse_pe_list[0] + sse_pe_list[1] + sse_pe_list[2] + sse_pe_list[3];
//                     _mm256_store_pd(sse_fx_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_xdiff), sse_vmag_sub_rlen_div_vmag));
//                     fx[nx * n + ny] += sse_fx_list[0] + sse_fx_list[1] + sse_fx_list[2] + sse_fx_list[3];
//                     _mm256_store_pd(sse_fy_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_ydiff), sse_vmag_sub_rlen_div_vmag));
//                     fy[nx * n + ny] += sse_fy_list[0] + sse_fy_list[1] + sse_fy_list[2] + sse_fy_list[3];
//                     _mm256_store_pd(sse_fz_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_zdiff), sse_vmag_sub_rlen_div_vmag));
//                     fz[nx * n + ny] += sse_fz_list[0] + sse_fz_list[1] + sse_fz_list[2] + sse_fz_list[3];
//                 }
//                 for (dy; dy < MIN(ny + delta + 1, n); dy++)
//                 {
//                     // compute reference distance
//                     rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) * sep;
//                     // compute actual distance
//                     xdiff = x[dx * n + dy] - x[nx * n + ny];
//                     ydiff = y[dx * n + dy] - y[nx * n + ny];
//                     zdiff = z[dx * n + dy] - z[nx * n + ny];
//                     vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
//                     // potential energy and force
//                     pe += fcon * (vmag - rlen) * (vmag - rlen);
//                     fx[nx * n + ny] += fcon * xdiff * (vmag - rlen) / vmag;
//                     fy[nx * n + ny] += fcon * ydiff * (vmag - rlen) / vmag;
//                     fz[nx * n + ny] += fcon * zdiff * (vmag - rlen) / vmag;
//                 }
//             }

//             // nx + 1 <= dx < MIN(nx + delta + 1, n);
//             for (dx = nx + 1; dx < MIN(nx + delta + 1, n); dx++)
//             {
//                 double sse_dx_rlen_list[4] = {(double)((nx - dx) * (nx - dx)), (double)((nx - dx) * (nx - dx)), (double)((nx - dx) * (nx - dx)), (double)((nx - dx) * (nx - dx))};
//                 __m256d sse_dx_rlen = _mm256_load_pd(sse_dx_rlen_list);

//                 // MAX(ny - delta, 0) < dy < MIN(ny + delta + 1, n)
//                 for (dy = MAX(ny - delta, 0); dy < MIN(ny + delta + 1, n) - 4; dy+=4)
//                 {
//                     // compute reference distance
//                     double sse_dy_rlen_list[4] = {(double)((ny - dy) * (ny - dy)), (double)((ny - (dy + 1)) * (ny - (dy + 1))), (double)((ny - (dy + 2)) * (ny - (dy + 2))), (double)((ny - (dy + 3)) * (ny - (dy + 3)))};
//                     __m256d sse_dy_rlen = _mm256_load_pd(sse_dy_rlen_list);
//                     __m256d sse_rlen = _mm256_mul_pd(_mm256_sqrt_pd(_mm256_add_pd(sse_dx_rlen, sse_dy_rlen)), sse_sep);

//                     // compute actual distance
//                     __m256d sse_dx = _mm256_load_pd(&x[dx * n + dy]);
//                     __m256d sse_dy = _mm256_load_pd(&y[dx * n + dy]);
//                     __m256d sse_dz = _mm256_load_pd(&z[dx * n + dy]);
//                     __m256d sse_xdiff = _mm256_sub_pd(sse_dx, sse_nx);
//                     __m256d sse_ydiff = _mm256_sub_pd(sse_dy, sse_ny);
//                     __m256d sse_zdiff = _mm256_sub_pd(sse_dz, sse_nz);
//                     __m256d sse_vmag = _mm256_sqrt_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(sse_xdiff, sse_xdiff), _mm256_mul_pd(sse_ydiff, sse_ydiff)), _mm256_mul_pd(sse_zdiff, sse_zdiff)));

//                     // potential energy and force
//                     __m256d sse_vmag_sub_rlen = _mm256_sub_pd(sse_vmag, sse_rlen);
//                     __m256d sse_vmag_sub_rlen_div_vmag = _mm256_div_pd(sse_vmag_sub_rlen, sse_vmag);
//                     _mm256_store_pd(sse_pe_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_vmag_sub_rlen), sse_vmag_sub_rlen));
//                     pe += sse_pe_list[0] + sse_pe_list[1] + sse_pe_list[2] + sse_pe_list[3];
//                     _mm256_store_pd(sse_fx_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_xdiff), sse_vmag_sub_rlen_div_vmag));
//                     fx[nx * n + ny] += sse_fx_list[0] + sse_fx_list[1] + sse_fx_list[2] + sse_fx_list[3];
//                     _mm256_store_pd(sse_fy_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_ydiff), sse_vmag_sub_rlen_div_vmag));
//                     fy[nx * n + ny] += sse_fy_list[0] + sse_fy_list[1] + sse_fy_list[2] + sse_fy_list[3];
//                     _mm256_store_pd(sse_fz_list, _mm256_mul_pd(_mm256_mul_pd(sse_fcon, sse_zdiff), sse_vmag_sub_rlen_div_vmag));
//                     fz[nx * n + ny] += sse_fz_list[0] + sse_fz_list[1] + sse_fz_list[2] + sse_fz_list[3];
//                 }
//                 for (dy; dy < MIN(ny + delta + 1, n); dy++)
//                 {
//                     // compute reference distance
//                     rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) * sep;
//                     // compute actual distance
//                     xdiff = x[dx * n + dy] - x[nx * n + ny];
//                     ydiff = y[dx * n + dy] - y[nx * n + ny];
//                     zdiff = z[dx * n + dy] - z[nx * n + ny];
//                     vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
//                     // potential energy and force
//                     pe += fcon * (vmag - rlen) * (vmag - rlen);
//                     fx[nx * n + ny] += fcon * xdiff * (vmag - rlen) / vmag;
//                     fy[nx * n + ny] += fcon * ydiff * (vmag - rlen) / vmag;
//                     fz[nx * n + ny] += fcon * zdiff * (vmag - rlen) / vmag;
//                 }
//             }
//         }
//     }
//     return 0.5 * pe;
// }

#include "./cloth_code.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>

void initMatrix(int n, double UNUSED(mass), double UNUSED(fcon),
                int UNUSED(delta), double UNUSED(grav), double sep,
                double rball, double offset, double UNUSED(dt), double **x,
                double **y, double **z, double **cpx, double **cpy,
                double **cpz, double **fx, double **fy, double **fz,
                double **vx, double **vy, double **vz, double **oldfx,
                double **oldfy, double **oldfz)
{
    int nx, ny;

    // Free any existing
    free(*x);
    free(*y);
    free(*z);
    free(*cpx);
    free(*cpy);
    free(*cpz);

    // allocate arrays to hold locations of nodes
    *x = (double *)malloc(n * n * sizeof(double));
    *y = (double *)malloc(n * n * sizeof(double));
    *z = (double *)malloc(n * n * sizeof(double));
    // This is for opengl stuff
    *cpx = (double *)malloc(n * n * sizeof(double));
    *cpy = (double *)malloc(n * n * sizeof(double));
    *cpz = (double *)malloc(n * n * sizeof(double));

    // initialize coordinates of cloth
    for (nx = 0; nx < n; nx++)
    {
        for (ny = 0; ny < n; ny++)
        {
            (*x)[n * nx + ny] = nx * sep - (n - 1) * sep * 0.5 + offset;
            (*z)[n * nx + ny] = rball + 1;
            (*y)[n * nx + ny] = ny * sep - (n - 1) * sep * 0.5 + offset;
            (*cpx)[n * nx + ny] = 0;
            (*cpz)[n * nx + ny] = 1;
            (*cpy)[n * nx + ny] = 0;
        }
    }

    // Throw away existing arrays
    free(*fx);
    free(*fy);
    free(*fz);
    free(*vx);
    free(*vy);
    free(*vz);
    free(*oldfx);
    free(*oldfy);
    free(*oldfz);
    // Alloc new
    // use calloc instead of malloc (calloc will intialize all the blocks into 0)
    *fx = (double *)calloc(n * n, sizeof(double));
    *fy = (double *)calloc(n * n, sizeof(double));
    *fz = (double *)calloc(n * n, sizeof(double));
    *vx = (double *)calloc(n * n, sizeof(double));
    *vy = (double *)calloc(n * n, sizeof(double));
    *vz = (double *)calloc(n * n, sizeof(double));
    *oldfx = (double *)malloc(n * n * sizeof(double));
    *oldfy = (double *)malloc(n * n * sizeof(double));
    *oldfz = (double *)malloc(n * n * sizeof(double));
}

void loopcode(int n, double mass, double fcon, int delta, double grav,
              double sep, double rball, double xball, double yball,
              double zball, double dt, double *x, double *y, double *z,
              double *fx, double *fy, double *fz, double *vx, double *vy,
              double *vz, double *oldfx, double *oldfy, double *oldfz,
              double *pe, double *ke, double *te)
{
    int i, j;
    double xdiff, ydiff, zdiff, vmag, damp, rball_div_vmag, cons_5;

    __m128d cons_3 = _mm_load1_pd(&dt);
    __m128d cons_4 = _mm_mul_pd(cons_3, _mm_div_pd(_mm_set1_pd(0.5), _mm_load1_pd(&mass)));
    for (j = 0; j < n; j++)
    {
        for (i = 0; i < (n / 2) * 2; i += 2)
        {
            // load two double precision
            __m128d sse_x = _mm_load_pd(&x[j * n + i]);
            __m128d sse_y = _mm_load_pd(&y[j * n + i]);
            __m128d sse_z = _mm_load_pd(&z[j * n + i]);
            __m128d sse_vx = _mm_load_pd(&vx[j * n + i]);
            __m128d sse_vy = _mm_load_pd(&vy[j * n + i]);
            __m128d sse_vz = _mm_load_pd(&vz[j * n + i]);
            __m128d sse_fx = _mm_load_pd(&fx[j * n + i]);
            __m128d sse_fy = _mm_load_pd(&fy[j * n + i]);
            __m128d sse_fz = _mm_load_pd(&fz[j * n + i]);

            _mm_store_pd(&oldfx[j * n + i], sse_fx);
            _mm_store_pd(&oldfy[j * n + i], sse_fy);
            _mm_store_pd(&oldfz[j * n + i], sse_fz);

            _mm_store_pd(&x[j * n + i], _mm_add_pd(sse_x, _mm_mul_pd(cons_3, _mm_add_pd(sse_vx, _mm_mul_pd(sse_fx, cons_4)))));
            _mm_store_pd(&y[j * n + i], _mm_add_pd(sse_y, _mm_mul_pd(cons_3, _mm_add_pd(sse_vy, _mm_mul_pd(sse_fy, cons_4)))));
            _mm_store_pd(&z[j * n + i], _mm_add_pd(sse_z, _mm_mul_pd(cons_3, _mm_add_pd(sse_vz, _mm_mul_pd(sse_fz, cons_4)))));
        }
        for (i = (n / 2) * 2; i < n; i++)
        {
            x[j * n + i] += dt * (vx[j * n + i] + dt * fx[j * n + i] * 0.5 / mass);
            oldfx[j * n + i] = fx[j * n + i];
            y[j * n + i] += dt * (vy[j * n + i] + dt * fy[j * n + i] * 0.5 / mass);
            oldfy[j * n + i] = fy[j * n + i];
            z[j * n + i] += dt * (vz[j * n + i] + dt * fz[j * n + i] * 0.5 / mass);
            oldfz[j * n + i] = fz[j * n + i];
        }
    }

    for (j = 0; j < n; j++)
    {
        for (i = 0; i < n; i++)
        {
            //	apply constraints - push cloth outside of ball
            xdiff = x[j * n + i] - xball;
            ydiff = y[j * n + i] - yball;
            zdiff = z[j * n + i] - zball;
            vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
            if (vmag < rball)
            {
                rball_div_vmag = rball / vmag;
                x[j * n + i] = xball + xdiff * rball_div_vmag;
                y[j * n + i] = yball + ydiff * rball_div_vmag;
                z[j * n + i] = zball + zdiff * rball_div_vmag;

                cons_5 = (vx[j * n + i] * xdiff + vy[j * n + i] * ydiff + vz[j * n + i] * zdiff) / (vmag * vmag);

                vx[j * n + i] = 0.1 * (vx[j * n + i] - xdiff * cons_5);
                vy[j * n + i] = 0.1 * (vy[j * n + i] - ydiff * cons_5);
                vz[j * n + i] = 0.1 * (vz[j * n + i] - zdiff * cons_5);
            }
        }
    }

    *pe = eval_pef(n, delta, mass, grav, sep, fcon, x, y, z, fx, fy, fz);

    // Add a damping factor to eventually set velocity to zero
    damp = 0.995;
    __m128d cons_6 = _mm_set1_pd(damp);
    for (j = 0; j < n; j++)
    {
        for (i = 0; i < (n / 2) * 2; i += 2)
        {
            __m128d sse_fx = _mm_load_pd(&fx[j * n + i]);
            __m128d sse_fy = _mm_load_pd(&fy[j * n + i]);
            __m128d sse_fz = _mm_load_pd(&fz[j * n + i]);
            __m128d sse_vx = _mm_load_pd(&vx[j * n + i]);
            __m128d sse_vy = _mm_load_pd(&vy[j * n + i]);
            __m128d sse_vz = _mm_load_pd(&vz[j * n + i]);
            __m128d sse_oldfx = _mm_load_pd(&oldfx[j * n + i]);
            __m128d sse_oldfy = _mm_load_pd(&oldfy[j * n + i]);
            __m128d sse_oldfz = _mm_load_pd(&oldfz[j * n + i]);

            _mm_store_pd(&vx[j * n + i], _mm_mul_pd(cons_6, _mm_add_pd(sse_vx, _mm_mul_pd(cons_4, _mm_add_pd(sse_fx, sse_oldfx)))));
            _mm_store_pd(&vy[j * n + i], _mm_mul_pd(cons_6, _mm_add_pd(sse_vy, _mm_mul_pd(cons_4, _mm_add_pd(sse_fy, sse_oldfy)))));
            _mm_store_pd(&vz[j * n + i], _mm_mul_pd(cons_6, _mm_add_pd(sse_vz, _mm_mul_pd(cons_4, _mm_add_pd(sse_fz, sse_oldfz)))));
        }
        for (i = (n / 2) * 2; i < n; i++)
        {
            vx[j * n + i] = (vx[j * n + i] + (fx[j * n + i] + oldfx[j * n + i]) * dt * 0.5 / mass) * damp;
            vy[j * n + i] = (vy[j * n + i] + (fy[j * n + i] + oldfy[j * n + i]) * dt * 0.5 / mass) * damp;
            vz[j * n + i] = (vz[j * n + i] + (fz[j * n + i] + oldfz[j * n + i]) * dt * 0.5 / mass) * damp;
        }
    }
    *ke = 0.0;
    for (j = 0; j < n; j++)
    {
        for (i = 0; i < n; i++)
        {

            *ke += vx[j * n + i] * vx[j * n + i] + vy[j * n + i] * vy[j * n + i] + vz[j * n + i] * vz[j * n + i];
        }
    }
    *ke = *ke / 2.0;
    *te = *pe + *ke;
}

double eval_pef(int n, int delta, double mass, double grav, double sep,
                double fcon, double *x, double *y, double *z, double *fx,
                double *fy, double *fz)
{
    double pe, rlen, xdiff, ydiff, zdiff, vmag;
    int nx, ny, dx, dy;

    pe = 0.0;
    __m128d sse_sep = _mm_set1_pd(sep);
    __m128d sse_fcon = _mm_set1_pd(fcon);
    // loop over particles
    for (nx = 0; nx < n; nx++)
    {
        for (ny = 0; ny < n; ny++)
        {
            fx[nx * n + ny] = 0.0;
            fy[nx * n + ny] = 0.0;
            fz[nx * n + ny] = -mass * grav;
            // loop over displacements
            __m128d sse_nx = _mm_set1_pd(x[nx * n + ny]);
            __m128d sse_ny = _mm_set1_pd(y[nx * n + ny]);
            __m128d sse_nz = _mm_set1_pd(z[nx * n + ny]);
            double sse_pe_list[2] = {0, 0};
            double sse_fx_list[2] = {0, 0};
            double sse_fy_list[2] = {0, 0};
            double sse_fz_list[2] = {0, 0};

            // dx = nx; MAX(ny - delta, 0 < dy < MIN(ny, n);
            for (dy = MAX(ny - delta, 0); dy < ny - 2; dy += 2)
            {
                // compute reference distance
                double sse_dy_rlen_list[2] = {(double)((ny - dy)), (double)((ny - (dy + 1)))};
                __m128d sse_dy_rlen = _mm_load_pd(sse_dy_rlen_list);
                __m128d sse_rlen = _mm_mul_pd(sse_dy_rlen, sse_sep);
                // compute actual distance
                __m128d sse_dx = _mm_load_pd(&x[nx * n + dy]);
                __m128d sse_dy = _mm_load_pd(&y[nx * n + dy]);
                __m128d sse_dz = _mm_load_pd(&z[nx * n + dy]);
                __m128d sse_xdiff = _mm_sub_pd(sse_dx, sse_nx);
                __m128d sse_ydiff = _mm_sub_pd(sse_dy, sse_ny);
                __m128d sse_zdiff = _mm_sub_pd(sse_dz, sse_nz);
                __m128d sse_vmag = _mm_sqrt_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(sse_xdiff, sse_xdiff), _mm_mul_pd(sse_ydiff, sse_ydiff)), _mm_mul_pd(sse_zdiff, sse_zdiff)));
                // potential energy and force
                __m128d sse_vmag_sub_rlen = _mm_sub_pd(sse_vmag, sse_rlen);
                __m128d sse_vmag_sub_rlen_div_vmag = _mm_div_pd(sse_vmag_sub_rlen, sse_vmag);
                _mm_store_pd(sse_pe_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_vmag_sub_rlen), sse_vmag_sub_rlen));
                pe += sse_pe_list[0] + sse_pe_list[1];
                _mm_store_pd(sse_fx_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_xdiff), sse_vmag_sub_rlen_div_vmag));
                fx[nx * n + ny] += sse_fx_list[0] + sse_fx_list[1];
                _mm_store_pd(sse_fy_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_ydiff), sse_vmag_sub_rlen_div_vmag));
                fy[nx * n + ny] += sse_fy_list[0] + sse_fy_list[1];
                _mm_store_pd(sse_fz_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_zdiff), sse_vmag_sub_rlen_div_vmag));
                fz[nx * n + ny] += sse_fz_list[0] + sse_fz_list[1];
            }
            for (dy; dy < ny; dy++)
            {
                // compute reference distance
                rlen = (double)((ny - dy)) * sep;
                // compute actual distance
                xdiff = x[nx * n + dy] - x[nx * n + ny];
                ydiff = y[nx * n + dy] - y[nx * n + ny];
                zdiff = z[nx * n + dy] - z[nx * n + ny];
                vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
                // potential energy and force
                pe += fcon * (vmag - rlen) * (vmag - rlen);
                fx[nx * n + ny] += fcon * xdiff * (vmag - rlen) / vmag;
                fy[nx * n + ny] += fcon * ydiff * (vmag - rlen) / vmag;
                fz[nx * n + ny] += fcon * zdiff * (vmag - rlen) / vmag;
            }

            // dx = nx; ny < dy < MIN(ny + delta + 1, n)
            for (dy = ny + 1; dy < MIN(ny + delta + 1, n) - 2; dy += 2)
            {
                // compute reference distance
                double sse_dy_rlen_list[2] = {(double)((ny - dy) * (ny - dy)), (double)((ny - (dy + 1)) * (ny - (dy + 1)))};
                __m128d sse_dy_rlen = _mm_load_pd(sse_dy_rlen_list);
                __m128d sse_rlen = _mm_mul_pd(_mm_sqrt_pd(sse_dy_rlen), sse_sep);
                // compute actual distance
                __m128d sse_dx = _mm_load_pd(&x[nx * n + dy]);
                __m128d sse_dy = _mm_load_pd(&y[nx * n + dy]);
                __m128d sse_dz = _mm_load_pd(&z[nx * n + dy]);
                __m128d sse_xdiff = _mm_sub_pd(sse_dx, sse_nx);
                __m128d sse_ydiff = _mm_sub_pd(sse_dy, sse_ny);
                __m128d sse_zdiff = _mm_sub_pd(sse_dz, sse_nz);
                __m128d sse_vmag = _mm_sqrt_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(sse_xdiff, sse_xdiff), _mm_mul_pd(sse_ydiff, sse_ydiff)), _mm_mul_pd(sse_zdiff, sse_zdiff)));
                // potential energy and force
                __m128d sse_vmag_sub_rlen = _mm_sub_pd(sse_vmag, sse_rlen);
                __m128d sse_vmag_sub_rlen_div_vmag = _mm_div_pd(sse_vmag_sub_rlen, sse_vmag);
                _mm_store_pd(sse_pe_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_vmag_sub_rlen), sse_vmag_sub_rlen));
                pe += sse_pe_list[0] + sse_pe_list[1];
                _mm_store_pd(sse_fx_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_xdiff), sse_vmag_sub_rlen_div_vmag));
                fx[nx * n + ny] += sse_fx_list[0] + sse_fx_list[1];
                _mm_store_pd(sse_fy_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_ydiff), sse_vmag_sub_rlen_div_vmag));
                fy[nx * n + ny] += sse_fy_list[0] + sse_fy_list[1];
                _mm_store_pd(sse_fz_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_zdiff), sse_vmag_sub_rlen_div_vmag));
                fz[nx * n + ny] += sse_fz_list[0] + sse_fz_list[1];
            }
            for (dy; dy < MIN(ny + delta + 1, n); dy++)
            {
                // compute reference distance
                rlen = sqrt((double)((ny - dy) * (ny - dy))) * sep;
                // compute actual distance
                xdiff = x[nx * n + dy] - x[nx * n + ny];
                ydiff = y[nx * n + dy] - y[nx * n + ny];
                zdiff = z[nx * n + dy] - z[nx * n + ny];
                vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
                // potential energy and force
                pe += fcon * (vmag - rlen) * (vmag - rlen);
                fx[nx * n + ny] += fcon * xdiff * (vmag - rlen) / vmag;
                fy[nx * n + ny] += fcon * ydiff * (vmag - rlen) / vmag;
                fz[nx * n + ny] += fcon * zdiff * (vmag - rlen) / vmag;
            }

            // MAX(nx - delta, 0) <= dx < nx;
            for (dx = MAX(nx - delta, 0); dx < nx; dx++)
            {
                double sse_dx_rlen_list[2] = {(double)((nx - dx) * (nx - dx)), (double)((nx - dx) * (nx - dx))};
                __m128d sse_dx_rlen = _mm_load_pd(sse_dx_rlen_list);

                // MAX(ny - delta, 0) < dy < MIN(ny + delta + 1, n)
                for (dy = MAX(ny - delta, 0); dy < MIN(ny + delta + 1, n) - 2; dy += 2)
                {
                    // compute reference distance
                    double sse_dy_rlen_list[2] = {(double)((ny - dy) * (ny - dy)), (double)((ny - (dy + 1)) * (ny - (dy + 1)))};
                    __m128d sse_dy_rlen = _mm_load_pd(sse_dy_rlen_list);
                    __m128d sse_rlen = _mm_mul_pd(_mm_sqrt_pd(_mm_add_pd(sse_dx_rlen, sse_dy_rlen)), sse_sep);

                    // compute actual distance
                    __m128d sse_dx = _mm_load_pd(&x[dx * n + dy]);
                    __m128d sse_dy = _mm_load_pd(&y[dx * n + dy]);
                    __m128d sse_dz = _mm_load_pd(&z[dx * n + dy]);
                    __m128d sse_xdiff = _mm_sub_pd(sse_dx, sse_nx);
                    __m128d sse_ydiff = _mm_sub_pd(sse_dy, sse_ny);
                    __m128d sse_zdiff = _mm_sub_pd(sse_dz, sse_nz);
                    __m128d sse_vmag = _mm_sqrt_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(sse_xdiff, sse_xdiff), _mm_mul_pd(sse_ydiff, sse_ydiff)), _mm_mul_pd(sse_zdiff, sse_zdiff)));

                    // potential energy and force
                    __m128d sse_vmag_sub_rlen = _mm_sub_pd(sse_vmag, sse_rlen);
                    __m128d sse_vmag_sub_rlen_div_vmag = _mm_div_pd(sse_vmag_sub_rlen, sse_vmag);
                    _mm_store_pd(sse_pe_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_vmag_sub_rlen), sse_vmag_sub_rlen));
                    pe += sse_pe_list[0] + sse_pe_list[1];
                    _mm_store_pd(sse_fx_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_xdiff), sse_vmag_sub_rlen_div_vmag));
                    fx[nx * n + ny] += sse_fx_list[0] + sse_fx_list[1];
                    _mm_store_pd(sse_fy_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_ydiff), sse_vmag_sub_rlen_div_vmag));
                    fy[nx * n + ny] += sse_fy_list[0] + sse_fy_list[1];
                    _mm_store_pd(sse_fz_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_zdiff), sse_vmag_sub_rlen_div_vmag));
                    fz[nx * n + ny] += sse_fz_list[0] + sse_fz_list[1];
                }
                for (dy; dy < MIN(ny + delta + 1, n); dy++)
                {
                    // compute reference distance
                    rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) * sep;
                    // compute actual distance
                    xdiff = x[dx * n + dy] - x[nx * n + ny];
                    ydiff = y[dx * n + dy] - y[nx * n + ny];
                    zdiff = z[dx * n + dy] - z[nx * n + ny];
                    vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
                    // potential energy and force
                    pe += fcon * (vmag - rlen) * (vmag - rlen);
                    fx[nx * n + ny] += fcon * xdiff * (vmag - rlen) / vmag;
                    fy[nx * n + ny] += fcon * ydiff * (vmag - rlen) / vmag;
                    fz[nx * n + ny] += fcon * zdiff * (vmag - rlen) / vmag;
                }
            }

            // nx + 1 <= dx < MIN(nx + delta + 1, n);
            for (dx = nx + 1; dx < MIN(nx + delta + 1, n); dx++)
            {
                double sse_dx_rlen_list[2] = {(double)((nx - dx) * (nx - dx)), (double)((nx - dx) * (nx - dx))};
                __m128d sse_dx_rlen = _mm_load_pd(sse_dx_rlen_list);

                // MAX(ny - delta, 0) < dy < MIN(ny + delta + 1, n)
                for (dy = MAX(ny - delta, 0); dy < MIN(ny + delta + 1, n) - 2; dy += 2)
                {
                    // compute reference distance
                    double sse_dy_rlen_list[2] = {(double)((ny - dy) * (ny - dy)), (double)((ny - (dy + 1)) * (ny - (dy + 1)))};
                    __m128d sse_dy_rlen = _mm_load_pd(sse_dy_rlen_list);
                    __m128d sse_rlen = _mm_mul_pd(_mm_sqrt_pd(_mm_add_pd(sse_dx_rlen, sse_dy_rlen)), sse_sep);

                    // compute actual distance
                    __m128d sse_dx = _mm_load_pd(&x[dx * n + dy]);
                    __m128d sse_dy = _mm_load_pd(&y[dx * n + dy]);
                    __m128d sse_dz = _mm_load_pd(&z[dx * n + dy]);
                    __m128d sse_xdiff = _mm_sub_pd(sse_dx, sse_nx);
                    __m128d sse_ydiff = _mm_sub_pd(sse_dy, sse_ny);
                    __m128d sse_zdiff = _mm_sub_pd(sse_dz, sse_nz);
                    __m128d sse_vmag = _mm_sqrt_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(sse_xdiff, sse_xdiff), _mm_mul_pd(sse_ydiff, sse_ydiff)), _mm_mul_pd(sse_zdiff, sse_zdiff)));

                    // potential energy and force
                    __m128d sse_vmag_sub_rlen = _mm_sub_pd(sse_vmag, sse_rlen);
                    __m128d sse_vmag_sub_rlen_div_vmag = _mm_div_pd(sse_vmag_sub_rlen, sse_vmag);
                    _mm_store_pd(sse_pe_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_vmag_sub_rlen), sse_vmag_sub_rlen));
                    pe += sse_pe_list[0] + sse_pe_list[1];
                    _mm_store_pd(sse_fx_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_xdiff), sse_vmag_sub_rlen_div_vmag));
                    fx[nx * n + ny] += sse_fx_list[0] + sse_fx_list[1];
                    _mm_store_pd(sse_fy_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_ydiff), sse_vmag_sub_rlen_div_vmag));
                    fy[nx * n + ny] += sse_fy_list[0] + sse_fy_list[1];
                    _mm_store_pd(sse_fz_list, _mm_mul_pd(_mm_mul_pd(sse_fcon, sse_zdiff), sse_vmag_sub_rlen_div_vmag));
                    fz[nx * n + ny] += sse_fz_list[0] + sse_fz_list[1];
                }
                for (dy; dy < MIN(ny + delta + 1, n); dy++)
                {
                    // compute reference distance
                    rlen = sqrt((double)((nx - dx) * (nx - dx) + (ny - dy) * (ny - dy))) * sep;
                    // compute actual distance
                    xdiff = x[dx * n + dy] - x[nx * n + ny];
                    ydiff = y[dx * n + dy] - y[nx * n + ny];
                    zdiff = z[dx * n + dy] - z[nx * n + ny];
                    vmag = sqrt(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff);
                    // potential energy and force
                    pe += fcon * (vmag - rlen) * (vmag - rlen);
                    fx[nx * n + ny] += fcon * xdiff * (vmag - rlen) / vmag;
                    fy[nx * n + ny] += fcon * ydiff * (vmag - rlen) / vmag;
                    fz[nx * n + ny] += fcon * zdiff * (vmag - rlen) / vmag;
                }
            }
        }
    }
    return 0.5 * pe;
}