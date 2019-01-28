#include <iostream>
#include "QPULib.h"

using namespace std;

// void gpu_transposition(Ptr<Int> A,Ptr<Int> B,Ptr<Int> C,Int m,Int n,Int k) {
//     Int qpuNums = numQPUs();

//     Int inc = 16;
//     Int ind = index();
//     Int inm = me()*k;

//     Ptr<Int> first_p = A+ind+inm;
//     Ptr<Int> first_q = B+ind;

//     Ptr<Int> p;
//     Ptr<Int> q;

//     Int x;
//     Int y;
//     Int sum;

//     Int output_offset = ind*n;

//     For(Int r=me(),r<m,r=r+qpuNums) 
//       For(Int c=0,c<n,c++)
//            p = first_p + ((r-me())*k);
//            q = first_q + (c*k);
//            gather(p);
//            gather(q);
//            sum = 0;
//            For(Int s=0,s<k,s=s+inc)
//               gather(p+inc);
//               gather(q+inc);
//               receive(x);
//               receive(y);
//               sum = sum + x*y;
//               p=p+inc;
//               q=q+inc;
//            End
//            receive(x);
//            receive(y);
//            // store(sum,C + ((r<<4)*n+c) + output_offset);
//            store(output_offset,C + output_offset);
//       End 
//     End 	
// }

void Init(SharedArray<int> &A,int m,int n) {
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      A[i*n+j] = i*n+j;
    }
  }

}

void cpu_gemm(SharedArray<int> &A,SharedArray<int>& B,int *C,int m,int n,int k) {
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      int sum = 0;
      for(int c=0;c<k;c++) {
        sum += A[i*k+c]*B[c*n+j];
      }
      C[i*n+j] = sum;
    }
  }
}

void gpu_test(Ptr<Int> C) {

    Int ind = index();
    Int a = *C;
    Int b = a;
    For(Int c=0,c<15,c=c+1)
      a = rotate(a,1);
      b = b + a;
    End
    store(b,C);
}



int main() {

  // auto GemmKernel = compile(gpu_transposition);
  // GemmKernel.setNumQPUs(1);

  auto GemmKernel = compile(gpu_test);
  GemmKernel.setNumQPUs(1);

  int m = 1;
  int k = 16;
  int n = 2;
  SharedArray<int> A(m*k),B(k*n),C(16);
  int *D = new int[m*n];
  int *E = new int[m*n];
  Init(A,m,k);
  Init(B,n,k);
  // GemmKernel(&A,&B,&C,m,n,k);
  for(int i=0;i<16;i++) {
    C[i] = i;
  }

  GemmKernel(&C);

  for(int i=0;i<16;i++) {
    cout<<C[i]<<" ";
  }
  cout<<endl;

  // cpu_gemm(A,B,D,m,n,k);


  // for(int i=0;i<m;i++) {
  //   for(int j=0;j<n;j++) {
  //     int sum = 0;
  //     for(int k=0;k<16;k++) {
  //       sum += C[(i+k)*n+j];
  //     }
  //     E[i*n+j] = sum;
  //   }
  // }

  // for(int i=0;i<m;i++) {
  //   for(int j=0;j<n;j++) {
  //     if(int(D[i*n+j])!=int(E[i*n+j])) {
  //       cout<<"error"<<endl;
  //       return 0;
  //     }
  //   }
  // }

  // cout<<"success"<<endl;

  return 0;
}