#include <iostream>
#include "QPULib.h"

using namespace std;

void gpu_transposition(Ptr<Float> A,Ptr<Float> B,Ptr<Float> C,Int m,Int n,Int k) {
    Int qpuNums = numQPUs();

    Int inc = 16;
    Int ind = index();
    Int inm = me()*k;

    Ptr<Float> first_p = A+ind+inm;
    Ptr<Float> first_q = B+ind;

    Ptr<Float> p;
    Ptr<Float> q;

    Float x;
    Float y;
    Float sum;

    Int output_offset = ind<<4;

    For(Int r=me(),r<m,r=r+qpuNums) 
      For(Int c=0,c<n,c++)
           p = first_p + ((r-me())*k);
           q = first_q + (c*k);
           gather(p);
           gather(q);
           sum = 0;
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
           store(sum,C + (r*n+c) + output_offset);
      End 
    End 	
}

void Init(SharedArray<float> &A,int m,int n) {
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      A[i*n+j] = i*n+j;
    }
  }

}

void cpu_gemm(SharedArray<float> &A,SharedArray<float>& B,float *C,int m,int n,int k) {
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

int main() {

  auto GemmKernel = compile(gpu_transposition);
  GemmKernel.setNumQPUs(12);

  int m = 10;
  int k = 32;
  int n = 10;
  SharedArray<float> A(m*k),B(k*n),C(m*n*16);
  float *D = new float[m*n];
  float *E = new float[m*n];
  Init(A,m,k);
  Init(B,k,n);
  GemmKernel(&A,&B,&C,m,n,k);
  cpu_gemm(A,B,D,m,n,k);


  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      float sum = 0;
      for(int k=0;k<16;k++) {
        sum += C[(i+k)*n+j];
      }
      E[i*n+j] = sum;
    }
  }

  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      if(int(D[i*n+j])!=int(E[i*n+j])) {
        cout<<"error"<<endl;
        break;
      }
    }
  }

  cout<<"success"<<endl;

  return 0;
}