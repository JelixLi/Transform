#include <iostream>
#include "QPULib.h"

using namespace std;

void gpu_gemm(Ptr<Float> A,Ptr<Float> B,Ptr<Float> C,Int m,Int n,Int k) {
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

int main() {

  auto GemmKernel = compile(gpu_transposition);
  GemmKernel.setNumQPUs(12);

  int m = 10;
  int k = 32;
  int n = 10;
  SharedArray<float> A(m*k),B(k*n),C(m*n);
  Init(A,m,k);
  Init(B,k,n);
  GemmKernel(A,B,C,m,n,k);
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
        cout<<C[i*n+j]<<" ";
    }
    cout<<endl;
  }
  return 0;
}