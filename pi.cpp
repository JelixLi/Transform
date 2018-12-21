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

    Int inc = (qpuNums<<4);
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
           p = first_p + (r*k);
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
           store(sum,C + ind + ((r*n+c)<<4));
      End 
    End 

}


#endif


void Init_Weight_Gpu(SharedArray<float> &A,int channels,int kernel_size,int output_num) {
  int size = channels*kernel_size*kernel_size;
  int padding = 16 - size % 16;

  int pos = 0;
  for(int i=0;i<output_num;i++) {
      for(int j=0;j<size;j++) {
          A[pos++] = pos;
      }

      for(int j=0;j<padding;j++) {
          A[pos++] = 0;
      }
  }
}

void Init_Weight_Cpu(SharedArray<float> &A,int channels,int kernel_size,int output_num) {
  int size = channels*kernel_size*kernel_size;

  int pos = 0;
  for(int i=0;i<output_num;i++) {
      for(int j=0;j<size;j++) {
          A[pos++] = pos;
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
    int output_num = 12;

    int channels = 1;
    int height = 4;
    int width = 4;
    int pad = 0;
    int stride = 1;
    int kernel_size = 3;

    int output_h = (height + 2 * pad - kernel_size) / stride + 1;
    int output_w = (width + 2 * pad - kernel_size) / stride + 1;

    int row_padding = 16 - (channels*kernel_size*kernel_size) % 16;

    int row_size = channels*kernel_size*kernel_size + row_padding;
    int col_size = output_h*output_w;


    int m = output_num;
    int n = col_size;
    int k = row_size;

    SharedArray<float> A(m*k),B(k*n),C(m*n*16);
    float D[m*n];
    float G[m*n];
    float E[k*n];

    Init_Weight_Gpu(A,channels,kernel_size,output_num);

    float *image = get_image(channels,height,width);

    transformToGpuFormat(B,image,height,width,channels,kernel_size,pad,stride);

    auto K=compile(gemm);
    K.setNumQPUs(1);

    clock_t start=clock();
    K(&A,&B,&C,m,n,k);
    clock_t end=clock();

    TransToCpuFormat(m*n*16,C,G); 

    printf("gpu_cost: %f\n",(end-start)/double(CLOCKS_PER_SEC)*1000);


    Init_Weight_Cpu(A,channels,kernel_size,output_num);

    im2col(image,channels,height,width,kernel_size,pad,stride,E);

    start=clock();
    cpu_gemm(A,E,D,m,n,k-row_padding);
    end=clock();


    printf("cpu_cost: %f\n",(end-start)/double(CLOCKS_PER_SEC)*1000);

    check(G,D,m,n);

    display_cpu(G,m,n);
    printf("\n");
    display_cpu(D,m,n);
    printf("\n");

}

