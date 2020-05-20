#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

	//registers
	__m256 zero = __mm256_load_ps(0);
	__m256 one = __mm256_load_ps(1);
	__m256 xvec = __mm256_load_ps(x);
	__m256 yvec = __mm256_load_ps(y);
	__m256 mvec = __mm256_load_ps(m);
	__m256 fx = __mm256_load_ps(0);
	__m256 fy = __mm256_load_ps(0);


  for(int i=0; i<N; i++) {
		

		//x[i] and y[i] registers
	  __m256 xi = _mm256_set1_ps(x[i]);
    __m256 yi = _mm256_set1_ps(y[i]);

		//calculation
		
		//float rx = x[i] - x[j];
		__m256 rx = _mm256_sub_ps(xi, xvec);
		//float ry = y[i] - y[j];
		__m256 ry = _mm256_sub_ps(yi, yvec);

		//if(i != j) 
		//mask not completed 
		__m256 m = _mm256_cmp_ps(rx, zero, _CMP_EQ_OQ);
		//float r = std::sqrt(rx * rx + ry * ry);
		//calculate rx squared
		__m256 rx2 = _mm256_mul_ps(rx, rx);
		__m256 ry2 = _mm256_mul_ps(ry, ry);
		// add them up
		__m256 sum = _mm256_add_ps(rx2, ry2);
		// square root
		__m256 r = _mm256_sqrt_ps(sum);


		//fx[i] -= rx * m[j] / (r * r * r);
		// r cubed
		__m256 r2 = _mm256_mul_ps(r,r);
		__m256 r3 = _mm256_mul_ps(r2,r);
		//rx * m[j] / (r ^ 3)
		__m256 lh = _mm256_mul_ps(rx, mvec);
		__m256 rh = _mm256_div_ps(lh, r3);
		fxvec = _mm256_mul_ps(fxvec, rh);
		_mm256_store_ps(fx, fxvec);

		//fx[i] -= rx * m[j] / (r * r * r);
		lh = _mm256_mul_ps(ry, mvec);
		rh = _mm256_div_ps(lh, r3);
		fyvec = _mm256_mul_ps(fyvec, rh);
		_mm256_store_ps(fy, fyvec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
