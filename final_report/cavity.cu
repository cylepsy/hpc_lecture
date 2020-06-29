// nsc.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include <fstream>

using namespace std;
const int nx = 41;
const int ny = 41;
const int nt = 500;
const int nit = 50;
const int ab = 2;

double dx = 2.0 / ((double) nx - 1.0);
double dy = 2.0 / ((double) ny - 1.0);
double rho = 1.0;
double nu = 0.1;
double dt = 0.001;
const int M = 128;
double u[ny * nx], v[ny * nx], p[ny * nx], b[ny * nx];
double * un = (double * ) malloc(nx * ny * sizeof(double));
double * vn = (double * ) malloc(nx * ny * sizeof(double));

__global__ void init() {
  for (int i = 0; i < ny; i++)
    for (int j = 0; j < nx; j++) {
      int k = i * nx + j;
      u[k] = 0;
      v[k] = 0;
      p[k] = 0;
      b[k] = 0;
    }
  //	std::cout << "init done" << std:endl;
}
__global__ void bulid_up_b(double * b, double rho, double dt, double * u, double * v, double dx, double dy) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = blockIdx.y;
  int index = idy * nx + idx;

  b[index] = rho * (1 / dt * ((u[index + 1] - u[index - 1]) / (2 * dx) + (v[index + nx] - v[index - nx]) / (2 * dy)) -
    pow((u[index + 1] - u[index - 1]) / (2 * dx), 2.0) -
    2 * ((u[(i + 1) * nx + j] - u[(i - 1) * nx + j]) / (2 * dy) *
      (v[index + 1] - v[index - 1]) / (2 * dx)) -
    pow((v[(i + 1) * nx + j] - v[(i) * nx + j]) / (2 * dy), 2.0));

}

__global__ void pressure_poisson(double * p, double dx, double dy, double * b) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = blockIdx.y;
  int index = idy * nx + idx;
  double * pn = (double * ) malloc(nx * ny * sizeof(double));

  for (int i = 0; i < (nx * ny); i++)
    pn[i] = p[i];
  if (idx > 0 && idx < nx - 1 && idy > 0 && idy < ny - 1) {
    p[index] = (((pn[index + 1] + pn[index - 1]) * pow(dy, 2) +
        (pn[index + nx] + pn[index - nx]) * pow(dx, 2)) /
      (2 * (pow(dx, 2) + pow(dy, 2))) -
      pow(dx, 2) * pow(dy, 2) / (2 * (pow(dx, 2) + pow(dy, 2))) *
      b[index]);
    if (idx == 0 && idy < ny) {
      p[index] = p[index + 1]; // dp/dx = 0 at left boundary
    } else if (idx == nx - 1 && idy < ny) {
      p[index] = p[index - 1]; // dp/dx = 0 at right boundary
    } else if (idy == 0 && idx < nx) {
      p[index] = p[index + nx]; // dp/dy = 0 at bottom boundary
    } else if (idy == ny - 1 && idx < nx) {
      p[index] = 0.0; // p  = 0 at top boundary
    }

  }
}
void cavity_flow(double * u, double * v, double dt, double dx, double dy,
  double * p, double rho, double nu, double * b) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int idy = blockIdx.y;
  int index = idy * nx + idx;

  for (int k = 0; k < nt; k++) {
    for (int i = 0; i < nx * ny; i++) {
      un[i] = u[i];
      vn[i] = v[i];
    }
  }
  if (idx > 0 && idx < nx - 1 && idy > 0 && idy < ny - 1) {
    u[index] = un[index] -
      un[index] * dt / dx * (un[index] - un[index - 1]) -
      vn[index] * dt / dy * (un[index] - un[index - nx]) -
      dt / (2 * rho * dx) * (p[index + 1] - p[index - 1]) +
      nu * (dt / pow(dx, 2.0) * (un[index + 1] - 2 * un[index] + un[index - 1]) +
        dt / pow(dy, 2.0) * (un[index + nx] - 2 * un[index] + un[index - nx]));
    v[index] = vn[index] -
      un[index] * dt / dx * (vn[index] - vn[index - 1]) -
      vn[index] * dt / dy * (vn[index] - vn[index - nx]) -
      dt / (2 * rho * dy) * (p[index + nx] - p[index - nx]) +
      nu * (dt / pow(dx, 2.0) * (vn[index + 1] - 2 * vn[index] + vn[index - 1]) +
        dt / pow(dy, 2.0) * (vn[index + nx] - 2 * vn[index] + vn[index - xn]));
  }
  if (idx < nx) {
    u[0 * nx + idx] = 0.0;
    u[(nx - 1) * nx + idx] = 1.0;
    v[0 * nx + idx] = 0.0;
    v[(nx - 1) * nx + idx] = 0.0;
  }

  if (idy < ny) {
    u[idy * nx + 0] = 0.0;
    u[idy * nx + nx - 1] = 0.0;
    v[idy * nx + 0] = 0.0;
    v[idy * nx + nx - 1] = 0.0;
  }

}

int main() {
  init();

  double * u, * v, * p, * uo, * vo, * po, * b;
  int size = ny * nx * sizeof(double);
  cudaMallocManaged( & u, size);
  cudaMallocManaged( & v, size);
  cudaMallocManaged( & p, size);
  cudaMallocManaged( & uo, size);
  cudaMallocManaged( & vo, size);
  cudaMallocManaged( & po, size);
  cudaMallocManaged( & b, size);

  dim3 block(M, 1);
  dim3 grid((nx + M - 1) / M, ny);

  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      u[j * nx + i] = 0;
      v[j * nx + i] = 0;
      p[j * nx + i] = 0;
      po[j * nx + i] = 0;
      b[j * nx + i] = 0;
    }

    for (int n = 0; n <= nt; n++) {
      bulid_up_b << < grid, block >>> (b, rho, dt, u, v, dx, dy);
      cudaDeviceSynchronize();
      pressure_poisson << < grid, block >>> (p, dx, dy, b);
      cudaDeviceSynchronize();
      cavity_flow << < grid, block >>> (u, v, dt, dx, dy, p, rho, nu, b);

    }
  }
  ofstream file("out.txt");
  for (int i = 0; i < ny; i++) {
    for (int j = 0; j < nx; j++)
      file << un[nx * i + j] << " ";
    file << "\n";
  }
  for (int i = 0; i < ny; i++) {
    for (int j = 0; j < nx; j++)
      file << vn[nx * i + j] << " ";
    file << "\n";
  }
  for (int i = 0; i < ny; i++) {
    for (int j = 0; j < nx; j++)
      file << p[nx * i + j] << " ";
    file << "\n";
  }
  file.close();
  cudaFree(u);
  cudaFree(v);
  cudaFree(p);
  cudaFree(uo);
  cudaFree(vo);
  cudaFree(po);
  cudaFree(b);
}