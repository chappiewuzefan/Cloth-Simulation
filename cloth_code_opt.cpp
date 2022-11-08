#include "./cloth_code.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
    double xdiff, ydiff, zdiff, vmag, damp, rball_div_vmag, cons_2;

    double cons_1 = dt * 0.5 / mass;

    for (j = 0; j < n; j++)
    {
        for (i = 0; i < n; i++)
        {
            // update position as per MD simulation
            x[j * n + i] += dt * (vx[j * n + i] + fx[j * n + i] * cons_1);
            y[j * n + i] += dt * (vy[j * n + i] + fy[j * n + i] * cons_1);
            z[j * n + i] += dt * (vz[j * n + i] + fz[j * n + i] * cons_1);
            oldfx[j * n + i] = fx[j * n + i];
            oldfy[j * n + i] = fy[j * n + i];
            oldfz[j * n + i] = fz[j * n + i];

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

                cons_2 = (vx[j * n + i] * xdiff + vy[j * n + i] * ydiff + vz[j * n + i] * zdiff) / (vmag * vmag);

                vx[j * n + i] = 0.1 * (vx[j * n + i] - xdiff * cons_2);
                vy[j * n + i] = 0.1 * (vy[j * n + i] - ydiff * cons_2);
                vz[j * n + i] = 0.1 * (vz[j * n + i] - zdiff * cons_2);
            }
        }
    }

    *pe = eval_pef(n, delta, mass, grav, sep, fcon, x, y, z, fx, fy, fz);

    // Add a damping factor to eventually set velocity to zero
    damp = 0.995;
    double temp_ke = 0.0;
    for (j = 0; j < n; j++)
    {
        for (i = 0; i < n; i++)
        {
            vx[j * n + i] = (vx[j * n + i] + (fx[j * n + i] + oldfx[j * n + i]) * cons_1) * damp;
            vy[j * n + i] = (vy[j * n + i] + (fy[j * n + i] + oldfy[j * n + i]) * cons_1) * damp;
            vz[j * n + i] = (vz[j * n + i] + (fz[j * n + i] + oldfz[j * n + i]) * cons_1) * damp;
            temp_ke += vx[j * n + i] * vx[j * n + i] + vy[j * n + i] * vy[j * n + i] + vz[j * n + i] * vz[j * n + i];
        }
    }
    *ke = temp_ke / 2.0;
    *te = *pe + *ke;
}

double eval_pef(int n, int delta, double mass, double grav, double sep,
                double fcon, double *x, double *y, double *z, double *fx,
                double *fy, double *fz)
{
    double pe, rlen, xdiff, ydiff, zdiff, vmag;
    int nx, ny, dx, dy;

    pe = 0.0;

    // loop over particles
    for (nx = 0; nx < n; nx++)
    {
        for (ny = 0; ny < n; ny++)
        {
            fx[nx * n + ny] = 0.0;
            fy[nx * n + ny] = 0.0;
            fz[nx * n + ny] = -mass * grav;
            // loop over displacements

            // dx = nx; MAX(ny - delta, 0) < dy < MIN(ny, n);
            for (dy = MAX(ny - delta, 0); dy < ny; dy++)
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
            for (dy = ny + 1; dy < MIN(ny + delta + 1, n); dy++)
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
            for (dx = MAX(nx - delta, 0); dx < nx; dx++) {
                // MAX(ny - delta, 0) < dy < MIN(ny + delta + 1, n)
                for (dy = MAX(ny - delta, 0); dy < MIN(ny + delta + 1, n); dy++)
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
                // MAX(ny - delta, 0) < dy < MIN(ny + delta + 1, n)
                for (dy = MAX(ny - delta, 0); dy < MIN(ny + delta + 1, n); dy++)
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