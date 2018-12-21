#include <stdlib.h> 
#include <time.h> 
#include <stdio.h>
#include <iostream>
#include <map>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <math.h>


using namespace std;

#define GPU

#ifdef GPU
#include "QPULib.h"
#endif

#ifdef GPU




using namespace std;

bool equal(float a, float b) {
    const float EPSILON = 1e-5;
    if (fabsf(a - b) < EPSILON) {
        return true;
    }
    return false;

}

constexpr const int MC = 384;

constexpr const int KC = 384;

constexpr const int NC_ = 4096;

constexpr const int MR = 4;

constexpr const int NR = 4;

float A_[MC * KC] __attribute__ ((aligned (32)));

float B_[KC * NC_] __attribute__ ((aligned (32)));

float C_[MR * NR] __attribute__ ((aligned (32)));

float AB_[MR * NR] __attribute__ ((aligned (32)));


void pack_MRxk(int k, const float *A, int iNC_RowA, int iNC_ColA, float *buffer) {
int j, a2 = iNC_RowA, a3 = 2 * iNC_RowA, a4 = 3 * iNC_RowA;
for (j = 0; j < k; ++j) {
    // for (int i = 0; i < MR; ++i) {
    //     buffer[i] = A[i * iNC_RowA];
    // }
    buffer[0] = A[0];
    buffer[1] = A[a2];
    buffer[2] = A[a3];
    buffer[3] = A[a4];
    A += 1;
    buffer += MR;
}
}

void pack_A(int mc, int kc, const float *A, int iNC_RowA, int iNC_ColA, float *buffer) {
int mp = mc / MR;
int _mr = mc % MR;
int tmp1 = kc * MR;
int tmp2 = MR * iNC_RowA;
int i, j;

for (i = 0; i < mp; ++i) {
    pack_MRxk(kc, A, iNC_RowA, iNC_ColA, buffer);
    buffer += tmp1;
    A += tmp2;
    // buffer += kc * MR;
    // A += MR * iNC_RowA;
}
if (_mr > 0) {
    for (j = 0; j < kc; ++j) {
        for (i = 0; i < _mr; ++i) {
            buffer[i] = A[i * iNC_RowA];
        }
        for (i = _mr; i < MR; ++i) {
            buffer[i] = 0.0;
        }
        A += 1;
        buffer += MR;
    }
}
}

void pack_kxNR(int k, const float *B, int iNC_RowB, int iNC_ColB, float *buffer) {
int i, j;
for (i = 0; i < k; ++i) {
    for (j = 0; j < NR; ++j) {
        buffer[j] = B[j];
    }
    // float32x4_t bv = vld1q_f32(B);
    // vst1q_f32(buffer, bv);
    B += iNC_RowB;
    buffer += NR;
}
}

void pack_B(int kc, int NC_, const float *B, int iNC_RowB, int iNC_ColB, float *buffer) {
int np = NC_ / NR;
int _nr = NC_ % NR;
int tmp1 = kc * NR;
int i, j;

for (j = 0; j < np; ++j) {
    pack_kxNR(kc, B, iNC_RowB, iNC_ColB, buffer);
    B += NR;
    buffer += tmp1;
}
if (_nr > 0) {
    for (i = 0; i < kc; ++i) {
        for (j = 0; j < _nr; ++j) {
            buffer[j] = B[j];
        }
        for (j = _nr; j < NR; ++j) {
            buffer[j] = 0.0;
        }
        buffer += NR;
        B += iNC_RowB;
    }
}
}

void dgemm_micro_kernel(int kc, float alpha, const float *A, const float *B, float beta, float *C, int iNC_RowC,
                       int iNC_ColC) {
int i = 0;
int j = 0;
int l = 0;
for (l = 0; l < MR * NR; ++l) {
    AB_[l] = 0;
}
for (l = 0; l < kc; ++l) {
    for (j = 0; j < NR; ++j) {
        for (i = 0; i < MR; ++i) {
            AB_[i + j * MR] += A[i] * B[j];
        }
    }
    A += MR;
    B += NR;
}

if (equal(beta, 0.0)) {
    for (j = 0; j < NR; ++j) {
        for (i = 0; i < MR; ++i) {
            C[i * iNC_RowC + j * iNC_ColC] = 0.0;
        }
    }
} else if (!equal(beta, 1.0)) {
    for (j = 0; j < NR; ++j) {
        for (i = 0; i < MR; ++i) {
            C[i * iNC_RowC + j * iNC_ColC] *= beta;
        }
    }
}

if (!equal(alpha, 1.0)) {
    for (j = 0; j < NR; ++j) {
        for (i = 0; i < MR; ++i) {
            C[i * iNC_RowC + j * iNC_ColC] += alpha * AB_[i + j * MR];
        }
    }
} else {
    for (j = 0; j < NR; ++j) {
        for (i = 0; i < MR; ++i) {
            C[i * iNC_RowC + j * iNC_ColC] += AB_[i + j * MR];
        }
    }
}
}


void dgeaxpy(int m, int n, float alpha, const float *X, int iNC_RowX, int iNC_ColX, float *Y, int iNC_RowY,
                 int iNC_ColY) {
int i, j;
if (!equal(alpha, 1.0)) {
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i) {
            Y[i * iNC_RowY + j] += alpha * X[i + j * iNC_ColX];
        }
    }
} else {
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i) {
            Y[i * iNC_RowY + j] += X[i + j * iNC_ColX];
        }
    }
}
}

void dgescal(int m, int n, float alpha, float *X, int iNC_RowX, int iNC_ColX) {
int i, j;
if (!equal(alpha, 0.0)) {
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            X[i * iNC_RowX + j] *= alpha;
        }
    }
} else {
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            X[i * iNC_RowX + j] = 0.0;
        }
    }
}
}

void dgemm_macro_kernel(int mc, int NC_, int kc, float alpha, float beta, float *C, int iNC_RowC, int iNC_ColC) {
int mp = (mc + MR - 1) / MR;
int np = (NC_ + NR - 1) / NR;

int _mr = mc % MR;
int _nr = NC_ % NR;

int i, j;

for (j = 0; j < np; ++j) {
    int nr = (j != np - 1 || _nr == 0) ? NR : _nr;

    for (i = 0; i < mp; ++i) {
        int mr = (i != mp - 1 || _mr == 0) ? MR : _mr;

        if (mr == MR && nr == NR) {
            dgemm_micro_kernel(kc, alpha, &A_[i * kc * MR], &B_[j * kc * NR], beta, &C[i * MR * iNC_RowC + j * NR], iNC_RowC, iNC_ColC);
        } else {
            dgemm_micro_kernel(kc, alpha, &A_[i * kc * MR], &B_[j * kc * NR], 0.0, C_, 1, MR);
            dgescal(mr, nr, beta, &C[i * MR * iNC_RowC + j * NR], iNC_RowC, iNC_ColC);
            dgeaxpy(mr, nr, 1.0, C_, 1, MR, &C[i * MR * iNC_RowC + j * NR], iNC_RowC, iNC_ColC);
        }
    }
}
}

void dgemm_nn(int m, int n, int k, float alpha, const float *A, int iNC_RowA, int iNC_ColA, const float *B, int iNC_RowB, int iNC_ColB, float beta, float *C, int iNC_RowC, int iNC_ColC) {
int mb = (m + MC - 1) / MC;
int nb = (n + NC_ - 1) / NC_;
int kb = (k + KC - 1) / KC;

int _mc = m % MC;
int _NC_ = n % NC_;
int _kc = k % KC;

int mc, NC_, kc;
int i, j, l;

float _beta;

if (equal(alpha, 0.0) ||  k == 0) {
    dgescal(m, n, beta, C, iNC_RowC, iNC_ColC);
    return;
}

for (j = 0; j < nb; ++j) {
    NC_ = (j != nb - 1 || _NC_ == 0) ? NC_ : _NC_;

    for (l = 0; l < kb; ++l) {
        kc = (l != kb - 1 || _kc == 0) ? KC : _kc;
        _beta = (l == 0) ? beta : 1.0;

        pack_B(kc, NC_, &B[l * KC * iNC_RowB + j * NC_], iNC_RowB, iNC_ColB, B_);

        for (i = 0; i < mb; ++i) {
            mc = (i != mb - 1 || _mc == 0) ? MC : _mc;

            pack_A(mc, kc, &A[i * MC * iNC_RowA + l * KC], iNC_RowA, iNC_ColA, A_);

            dgemm_macro_kernel(mc, NC_, kc, alpha, _beta, &C[i * MC * iNC_RowC + j * NC_], iNC_RowC, iNC_ColC);
        }
    }
}
}

void sgemm(int m, int n, int k, const float *A, const float *B, float *C) {
dgemm_nn(m, n, k, 1, A, k, 1, B, n, 1, 0, C, n, 1);
}

void sgemm(int m, int n, int k, const float *A, const float *B, float *C, float alpha, float beta) {
dgemm_nn(m, n, k, alpha, A, k, 1, B, n, 1, beta, C, n, 1);
}



inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}



void im2col(const float *data_im, const int channels, const int height,
            const int width, const int kernel_size,
            const int pad, const int stride, float *data_col) {
    const int output_h = (height + 2 * pad - kernel_size) / stride + 1;
    const int output_w = (width + 2 * pad - kernel_size) / stride + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                int input_row = -pad + kernel_row;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -pad + kernel_col;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col += stride;
                        }
                    }
                    input_row += stride;
                }
            }
        }
    }
}


void transformToGpuFormat(
    SharedArray<float> &_shared_array_buffer,
    const float *input_data_buffer,
    int input_height,
    int input_width,
    int input_channel,
    int kernel_size,
    int pad,
    int stride) {

    SharedArray<float> &output_data=_shared_array_buffer;

    const int output_h = (input_height + 2 * pad - kernel_size) / stride + 1;
    const int output_w = (input_width + 2 * pad - kernel_size) / stride + 1;

    const int row_padding = 16 - (input_channel*kernel_size*kernel_size) % 16;

    const float *input_data;

    int array_pos = 0;

    for(int row=-pad;row<input_height+pad-kernel_size+1;row+=stride) {
        for(int col=-pad;col<input_width+pad-kernel_size+1;col+=stride) {

            for(int chan=0;chan<input_channel;chan++) {

                input_data = input_data_buffer + chan*input_height*input_width;

                for(int kernel_row=0;kernel_row<kernel_size;kernel_row++) {
                    for(int kernel_col=0;kernel_col<kernel_size;kernel_col++) {

                        int new_row=row+kernel_row;
                        int new_col=col+kernel_col;

                        if(is_a_ge_zero_and_a_lt_b(new_row,input_height)&&is_a_ge_zero_and_a_lt_b(new_col,input_width)) {
                            output_data[array_pos++] = input_data[new_row*input_width+new_col];
                        } else {
                            output_data[array_pos++] = 0;
                        }               
                    }
                }

            }

            for(int i=0;i<row_padding;i++) {
                 output_data[array_pos++] = 0;
            }

        }
    }    

}




void TransToCpuFormat(
    int data_num,
    SharedArray<float> &gpu_vec_data,
    float *cpu_data) {

    int sum=0;
    for(int i=1;i<=data_num;i++) {
        if(i%16==0) {
            *cpu_data++ = sum;
            sum = 0;
        }

        sum += gpu_vec_data[i];
    }
}

void gemm(Ptr<Float> A,Ptr<Float> B,Ptr<Float> C,Int m,Int n,Int k) {

    Int qpuNums = numQPUs();

    Int iNC_ = 16;
    Int ind = index();
    Int inm = me()*k;

    Ptr<Float> first_p = A+ind+inm;
    Ptr<Float> first_q = B+ind;

    Ptr<Float> p;
    Ptr<Float> q;

    Float x;
    Float y;
    Float sum;

    For(Int r=me(),r<m,r=r+qpuNums) 
      For(Int c=0,c<n,c++)
           p = first_p + ((r-me())*k);
           q = first_q + (c*k);
           gather(p);
           gather(q);
           sum = 0;
           For(Int s=0,s<k,s=s+iNC_)
              gather(p+iNC_);
              gather(q+iNC_);
              receive(x);
              receive(y);
              sum = sum + x*y;
              p=p+iNC_;
              q=q+iNC_;
           End
           receive(x);
           receive(y);
           store(sum,C + ind + ((r*n+c)<<4));
      End 
    End 

}


#endif


void Init_Weight_Gpu(SharedArray<float> &A,int channels,int kernel_size,int output_num) {
  int size = channels*kernel_size*kernel_size;
  int padding = 16 - size % 16;

  int pos = 0;
  int num = 0;
  for(int i=0;i<output_num;i++) {
      for(int j=0;j<size;j++) {
          A[pos++] = num++;
      }

      for(int j=0;j<padding;j++) {
          A[pos++] = 0;
      }
  }
}

void Init_Weight_Cpu(SharedArray<float> &A,int channels,int kernel_size,int output_num) {
  int size = channels*kernel_size*kernel_size;

  int pos = 0;
  int num = 0;
  for(int i=0;i<output_num;i++) {
      for(int j=0;j<size;j++) {
          A[pos++] = num++;
      }
  }
}

void cpu_gemm(SharedArray<float> &A,float *B,float *C,int m,int n,int k) {
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


void check(float *A,float *B,int m,int n) {
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
      if(int(A[i*n+j])!=int(B[i*n+j])) {
          printf("error\n");
          return;
      }
    }
  }
  printf("success\n");
}

void display(SharedArray<float> &A,int m,int n) {
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
          printf("%f ",A[i*n+j]);  
    }
    printf("\n");
  } 
}

void display_cpu(float *A,int m,int n) {
  for(int i=0;i<m;i++) {
    for(int j=0;j<n;j++) {
          printf("%f ",A[i*n+j]);  
    }
    printf("\n");
  }  
}


float *get_image(int channels,int height,int width) {
    float *image = new float[channels*height*width];
    for(int i=0;i<channels*height*width;i++) {
      image[i] = i;
    }
    return image;
}

int main() {
    int output_num = 196;

    int channels = 2;
    int height = 2;
    int width = 2;
    int pad = 0;
    int stride = 1;
    int kernel_size = 1;

    int output_h = (height + 2 * pad - kernel_size) / stride + 1;
    int output_w = (width + 2 * pad - kernel_size) / stride + 1;

    int row_padding = 16 - (channels*kernel_size*kernel_size) % 16;

    int row_size = channels*kernel_size*kernel_size + row_padding;
    int col_size = output_h*output_w;


    int m = output_num;
    int n = col_size;
    int k = row_size;

    SharedArray<float> A(m*k),B(k*n),C(m*n*16);
    float F[m*k];
    float D[m*n];
    float G[m*n];
    float E[k*n];

    Init_Weight_Gpu(A,channels,kernel_size,output_num);

    float *image = get_image(channels,height,width);

    auto K=compile(gemm);
    K.setNumQPUs(12);

    clock_t start=clock();
    transformToGpuFormat(B,image,height,width,channels,kernel_size,pad,stride);
    K(&A,&B,&C,m,n,k);
    TransToCpuFormat(m*n*16,C,G); 
    clock_t end=clock();

    printf("gpu_cost: %f\n",(end-start)/double(CLOCKS_PER_SEC)*1000);


    Init_Weight_Cpu(A,channels,kernel_size,output_num);

    for(int i=0;i<m*k;i++) {
      F[i] = A[i];
    }

    start=clock();
    // cpu_gemm(A,E,D,m,n,k-row_padding);
    im2col(image,channels,height,width,kernel_size,pad,stride,E);
    sgemm(m,n,k-row_padding,F,E,D);
    end=clock();


    printf("cpu_cost: %f\n",(end-start)/double(CLOCKS_PER_SEC)*1000);

    check(G,D,m,n);

    display_cpu(G,m,n);
    printf("\n");
    display_cpu(D,m,n);
    printf("\n");

}

