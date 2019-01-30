#include <stdlib.h> 
#include <time.h> 
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <typeinfo>

#include "QPULib.h"

using namespace std;

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void im2col(const float *data_im, const int channels, const int height,
            const int width, const int kernel_size,
            const int pad, const int stride, float *data_col) {
    const int output_h = (height + 2 * pad - kernel_size) / stride + 1;
    const int output_w = (width + 2 * pad - kernel_size) / stride + 1;
    const int channel_size = height * width;

    register float *data_col_ptr = data_col;
    register const float *data_im_ptr = data_im;

    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                int input_row = -pad + kernel_row;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        int n = output_w/4;
                        int _n = output_w/4;
                        // for (int output_cols = output_w; output_cols; output_cols--) {
                        //     *(data_col++) = 0;
                        // }
                        for(int i=0;i<n;i+=4) {
                           *(data_col++) = 0;
                           *(data_col++) = 0;
                           *(data_col++) = 0;
                           *(data_col++) = 0;
                        }
                        for(int i=0;i<n;i++) {
                           *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -pad + kernel_col;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                // *(data_col++) = data_im[input_row * width + input_col];
                                *(data_col++) = *(data_im + input_row * width + input_col);
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

void get_gpu_format_weight(SharedArray<float>& gpu_format_weight,float *weight,int kernel_size) {
  int n = kernel_size*kernel_size;
  int pos = 0;
  float tmp;
  for(int i=0;i<n;i++) {
    tmp = *weight;
    for(int j=0;j<16;j++) {
      gpu_format_weight[pos++] = tmp;
    }
    weight++;
  }
}

void loadSingleImageIntoGpu(
  SharedArray<float>& gpu_format_data,
  int offset,
  float *col_data,
  int kernel_size,
  int height,
  int width,
  int pad,
  int stride) {

  int output_h = (height + 2 * pad - kernel_size) / stride + 1; 
  int output_w = (width + 2 * pad - kernel_size) / stride + 1; 

  int row_size = kernel_size*kernel_size;
  int col_size = output_w*output_h;

  int r = col_size / 16;
  int _r = col_size % 16;

  float *col_data_ptr;
  int pos = offset;

  for(int i=0;i<r;i++) {
    for(int j=0;j<row_size;j++) {
        col_data_ptr = col_data + col_size*j + i*16;
        for(int k=0;k<16;k++) {
            gpu_format_data[pos++] = *col_data_ptr++; 
        }
    }
  }

  for(int j=0;j<row_size;j++) {
      col_data_ptr = col_data + col_size*j + r*16;
      for(int k=0;k<_r;k++) {
          gpu_format_data[pos++] = *col_data_ptr++; 
      }
      pos += 16 - _r; 
  }


}


void loadMultiImagesIntoGpu(  
  SharedArray<float>& gpu_format_data,
  float *col_data,
  int kernel_size,
  int channels,
  int height,
  int width,
  int pad,
  int stride) {

  int output_h = (height + 2 * pad - kernel_size) / stride + 1; 
  int output_w = (width + 2 * pad - kernel_size) / stride + 1; 

  int row_size = kernel_size*kernel_size;
  int col_size = output_w*output_h;

  int col_buffer_offset = row_size*col_size;
  int gpu_buffer_offset = (col_size / 16 + 1)*16*row_size;

  for(int i=0;i<channels;i++) {
    loadSingleImageIntoGpu(
      gpu_format_data,
      i*gpu_buffer_offset,
      col_data+i*col_buffer_offset,
      kernel_size,
      height,
      width,
      pad,
      stride);
  }
}


void gpu_depthwise_gemm(Ptr<Float> A,Ptr<Float> B,Ptr<Float> C,Int kernel_num,Int block_num,Int block_size) {
    Int qpuNums = numQPUs();
    Int ind = index();
    Int inc = 16;
    Int inm = (me()<<4);
    Int output_offset;

    Ptr<Float> base_p;
    Ptr<Float> p;
    Ptr<Float> q;

    Float x;
    Float y;
    Float sum;

    For(Int r=me(),r<kernel_num,r=r+qpuNums) 
      base_p = A+ind+inm*block_size;
      q = B+ind+inm*block_size;
      output_offset = r*block_num*block_size;
      For(Int c=0,c<block_num,c++)
           p = base_p;
           gather(p);
           gather(q);
           sum = 0;
           For(Int s=0,s<block_size,s++)
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
           store(sum,C + output_offset);
           output_offset = output_offset + inc;
      End 
    End   
}


int main() {

  int channels = 10;

  int height = 200;
  int width = 200;
  int kernel_size = 3;
  int pad = 1;
  int stride = 2;

  int output_h = (height + 2 * pad - kernel_size) / stride + 1; 
  int output_w = (width + 2 * pad - kernel_size) / stride + 1; 

  float *image = new float[height*width*channels];
  float *weight = new float[kernel_size*kernel_size*channels];

  // float *gpu_format_weight = new float[16*kernel_size*kernel_size*channels];
  SharedArray<float> gpu_format_weight(16*kernel_size*kernel_size*channels);

  get_gpu_format_weight(gpu_format_weight,weight,kernel_size);

  float *col_data = new float[kernel_size*kernel_size*output_w*output_h*channels];

  im2col(image,channels,height,width,kernel_size,pad,stride,col_data);

  // float *gpu_format_image = new float[channels*output_w*output_h*16];
  SharedArray<float> gpu_format_image(channels*output_w*output_h*16);
  SharedArray<float> gpu_output_buffer(channels*output_w*output_h*16);

  loadMultiImagesIntoGpu(gpu_format_image,col_data,kernel_size,channels,height,width,pad,stride);

  auto DepthwiseKernel = compile(gpu_depthwise_gemm);
  DepthwiseKernel.setNumQPUs(12);

  int n = output_w*output_h;
  int block_num = (n%16==0?n/16:n/16+1);

  DepthwiseKernel(
    gpu_format_weight,
    gpu_format_image,
    gpu_output_buffer,
    channels,
    block_num,
    (kernel_size*kernel_size));
}


