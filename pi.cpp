#include <stdlib.h> 
#include <time.h> 
#include <stdio.h>
#include <iostream>


using namespace std;

#define GPU

#ifdef GPU
#include "QPULib.h"
#endif

#ifdef GPU

void gemm(Ptr<Float> A,Ptr<Float> B,Ptr<Float> C,Int m,Int n,Int k) {

   Int qpuNums = numQPUs();

   Int inc=(qpuNums<<4);
   Int ind=index();
   Int inm=(me()<<4)*m;

   Ptr<Float> p=A+ind+inm;
   Ptr<Float> q=B+ind;

   Float x;
   Float y;
   Float sum;

   For(Int r=0,r<m,r=r+qpuNums) 
      For(Int c=0,c<n,c++)
           p = p + ((r*k)<<4);
           q = q + ((c*k)<<4);
           gather(p);
           gather(q);
           For(Int s=0,s<k,s=s+inc)
              gather(p+inc);
              gather(q+inc);
              receive(x);
              receive(y);
              sum = sum + x*y;
              p=p+inc;
              q=q+inc;
           End
           receive(x);
           receive(y);
           store(sum,C + ((r*m)<<4) + (n<<4));
      End 
   End 

}


#endif


void Init(SharedArray<float> input,int m,int n) {
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      input[i*n+j] = float(i*n+j);
    }
  }
}

void cpu_gemm(SharedArray<float> A,SharedArray<float> B,SharedArray<float> C,int m,int n,int k) {
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      float sum = 0;
      for(int c=0;c<k;c++) {
         sum += A[i*k+c]*B[c*n+j]; 
      }
      C[i*n+j] = sum;
    }
  }
}


void check(SharedArray<float> A,float *B,int m,int n) {
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      if(int(A[i*n+j])!=int(B[i*n+j])) {
          printf("error\n");
          break;
      }
    }
  }
  printf("success\n");
}

int main() {
   int m = 10;
   int n = 10;
   int k = 10;

   SharedArray<float> A(m*k*256),B(k*n*256),C(m*n*256);
   float D[m*16][n*16];

   Init(A,m*16,k*16);
   Init(B,k*16,n*16);

   auto K=compile(gemm);
   K.setNumQPUs(12);

   clock_t start=clock();
   K(&A,&B,&C,m,n,k);
   clock_t end=clock();

   printf("gpu_cost: %f\n",(end-start)/double(CLOCKS_PER_SEC)*1000);

   cpu_gemm(A,B,D,m,n,k);
   check(C,D,m,n);

}




 





