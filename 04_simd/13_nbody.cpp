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
<<<<<<< HEAD

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
=======
  __m256 zero = _mm256_setzero_ps();
  for(int i=0; i<N; i+=8) {
    __m256 xi = _mm256_load_ps(x+i);
    __m256 yi = _mm256_load_ps(y+i);
    __m256 fxi = zero;
    __m256 fyi = zero;
    for(int j=0; j<N; j++) {
      __m256 dx = _mm256_set1_ps(x[j]);
      __m256 dy = _mm256_set1_ps(y[j]);
      __m256 mj = _mm256_set1_ps(m[j]);
      __m256 r2 = zero;
      dx = _mm256_sub_ps(xi, dx);
      dy = _mm256_sub_ps(yi, dy);
      r2 = _mm256_fmadd_ps(dx, dx, r2);
      r2 = _mm256_fmadd_ps(dy, dy, r2);
      __m256 mask = _mm256_cmp_ps(r2, zero, _CMP_GT_OQ);
      __m256 invR = _mm256_rsqrt_ps(r2);
      invR = _mm256_blendv_ps(zero, invR, mask);
      mj = _mm256_mul_ps(mj, invR);
      invR = _mm256_mul_ps(invR, invR);
      mj = _mm256_mul_ps(mj, invR);
      fxi = _mm256_fmadd_ps(dx, mj, fxi);
      fyi = _mm256_fmadd_ps(dy, mj, fyi);
    }
    _mm256_store_ps(fx+i, fxi);
    _mm256_store_ps(fy+i, fyi);
>>>>>>> f743798ff25f63cf544466b630c34b35525ca76f
  }
  for(int i=0; i<N; i++)
    printf("%d %g %g\n",i,fx[i],fy[i]);
}
