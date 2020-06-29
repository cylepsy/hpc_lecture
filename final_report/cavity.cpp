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
double u[ny * nx], v[ny * nx], p[ny * nx], b[ny * nx];

void init() {
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
void bulid_up_b(double * b, double rho, double dt, double * u, double * v, double dx, double dy) {
  for (int i = 1; i < ny - 1; i++) {
    for (int j = 1; j < nx - 1; j++) {
      b[i * nx + j] = rho * (1 / dt * ((u[i * nx + j + 1] - u[i * nx + j - 1]) / (2 * dx) + (v[(i + 1) * nx + j] - v[(i - 1) * nx + j]) / (2 * dy)) -
        pow((u[i * nx + j + 1] - u[i * nx + j - 1]) / (2 * dx), 2.0) -
        2 * ((u[(i + 1) * nx + j] - u[(i - 1) * nx + j]) / (2 * dy) *
          (v[i * nx + j + 1] - v[i * nx + j - 1]) / (2 * dx)) -
        pow((v[(i + 1) * nx + j] - v[(i) * nx + j]) / (2 * dy), 2.0));
    }
  }
}

void pressure_poisson(double * p, double dx, double dy, double * b) {
  double * pn = (double * ) malloc(nx * ny * sizeof(double));
  for (int k = 0; k < nit; k++) {
    for (int i = 0; i < (nx * ny); i++)
      pn[i] = p[i];
    for (int i = 1; i < ny - 1; i++) {
      for (int j = 1; j < nx - 1; j++) {
        p[nx * i + j] = (((pn[nx * i + j + 1] + pn[nx * i + j - 1]) * pow(dy, 2) +
            (pn[nx * (i + 1) + j] + pn[nx * (i - 1) + j]) * pow(dx, 2)) /
          (2 * (pow(dx, 2) + pow(dy, 2))) -
          pow(dx, 2) * pow(dy, 2) / (2 * (pow(dx, 2) + pow(dy, 2))) *
          b[nx * i + j]);
      }
    }
    for (int i = 0; i < nx; i++) {
      p[i * nx + nx - 1] = p[i * nx + nx - 2];
      p[i] = p[nx + i];
      p[i * nx] = p[i * nx + 1];
      p[(ny - 1) * nx + i] = 0;
    }
  }
}

int main() {
  init();

  double * un = (double * ) malloc(nx * ny * sizeof(double));
  double * vn = (double * ) malloc(nx * ny * sizeof(double));
  for (int k = 0; k < nt; k++) {
    for (int i = 0; i < nx * ny; i++) {
      un[i] = u[i];
      vn[i] = v[i];
    }
  }

  for (int n = 0; n <= nt; n++) {
    bulid_up_b(b, rho, dt, u, v, dx, dy);
    pressure_poisson(p, dx, dy, b);

    for (int j = 1; j < ny - 1; j++) {
      for (int i = 1; i < nx - 1; i++) {
        u[j * nx + i] = un[j * nx + i] -
          un[j * nx + i] * dt / dx * (un[j * nx + i] - un[j * nx + i - 1]) -
          vn[j * nx + i] * dt / dy * (un[j * nx + i] - un[(j - 1) * nx + i]) -
          dt / (2 * rho * dx) * (p[j * nx + i + 1] - p[j * nx + i - 1]) +
          nu * (dt / pow(dx, 2.0) * (un[j * nx + i + 1] - 2 * un[j * nx + i] + un[j * nx + i - 1]) +
            dt / pow(dy, 2.0) * (un[(j + 1) * nx + i] - 2 * un[j * nx + i] + un[(j - 1) * nx + i]));
        v[j * nx + i] = vn[j * nx + i] -
          un[j * nx + i] * dt / dx * (vn[j * nx + i] - vn[j * nx + i - 1]) -
          vn[j * nx + i] * dt / dy * (vn[j * nx + i] - vn[(j - 1) * nx + i]) -
          dt / (2 * rho * dy) * (p[(j + 1) * nx + i] - p[(j - 1) * nx + i]) +
          nu * (dt / pow(dx, 2.0) * (vn[j * nx + i + 1] - 2 * vn[j * nx + i] + vn[j * nx + i - 1]) +
            dt / pow(dy, 2.0) * (vn[(j + 1) * nx + i] - 2 * vn[j * nx + i] + vn[(j - 1) * nx + i]));
      }
    }
    for (int i = 0; i < nx; i++) {
      u[i] = 0;
      u[i * nx] = 0;
      u[i * nx + nx - 1] = 0;
      v[i] = 0;
      v[i * nx] = 0;
      v[(ny - 1) * nx + i] = 0;
      v[i * nx + nx - 1] = 0;
    }
    for (int i = 0; i < nx; i++)
      u[(ny - 1) * nx + i] = 1;
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
}