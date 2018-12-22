#include <stdlib.h> 
#include <time.h> 
#include <stdio.h>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <math.h>


#define GPU


#ifdef GPU
#include "QPULib.h"
#endif

using namespace std;

#ifndef GPU

template<typename T>
class SharedArray {
public:
  SharedArray();
  SharedArray(int size);
  ~SharedArray();

  void alloc(int size);
  void dealloc();

  T &operator[](int id) {
    return data[id];
  }

  int getArraySize() const {
    return _size;
  }

private:
  T *data;
  int _size;
};


template<typename T>
SharedArray<T>::SharedArray():data(NULL){}


template<typename T>
SharedArray<T>::SharedArray(int size):_size(size) {
  alloc(size);
}


template<typename T>
SharedArray<T>::~SharedArray() {
  dealloc();
}

template<typename T>
void SharedArray<T>::alloc(int size) {
  data = new T[size];
}


template<typename T>
void SharedArray<T>::dealloc() {
  if(data) {
    delete [] data;
  }
}

#endif 


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
           store(sum,C + ind + ((r*n+c)<<4));
      End 
    End 	
}



auto GemmKernel = compile(gpu_gemm);


template<typename T>
class GManager {
public:
	GManager(int gpu_num);

	void LoadInputIntoGpu(
		SharedArray<T> &_shared_array_buffer,
		const T *input_data_buffer,
		int input_height,
		int input_width,
		int input_channel,
		int kernel_size,
		int pad,
		int stride);

	void LoadWeightIntoGpu(
		SharedArray<T> &_shared_array_buffer,
		const T *input_data_buffer,
		int data_size);

	void getOutputFromGpu(
		SharedArray<T> &_shared_array_buffer,
		T *output_data_buffer,
		int data_size);

	void gpu_gemm(Ptr<Float> A,Ptr<Float> B,Ptr<Float> C,Int m,Int n,Int k);

	void gpu_conv(
		T *weight,
		T *input,
		T *output_buffer,
		int height,
		int width,
		int channels,
		int kernel_size,
		int output_num,
		int pad,
		int stride);


	T *TransWeight2GpuFormat(T *original_data,int channels,int kernel_size,int output_num);

private:
	void Init_Gpu_Memory();

	void Init_Kernel(int gpu_num);


	inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
	    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
	}


	SharedArray<T> _gp_array[3];

};

template<typename T>
void GManager<T>::Init_Kernel(int gpu_num) {
    GemmKernel.setNumQPUs(gpu_num);
}


template<typename T>
void GManager<T>::getOutputFromGpu(
	SharedArray<T> &_shared_array_buffer,
	T *output_data_buffer,
	int data_size) {

    float sum=0;
    for(int i=0;i<data_size;i++) {
        sum += _shared_array_buffer[i];
        if((i+1)%16==0) {
            *output_data_buffer++ = sum;
            sum = 0;
        }
    }
}


template<typename T>
void GManager<T>::gpu_conv(
	T *weight,
	T *input,
	T *output,
	int height,
	int width,
	int channels,
	int kernel_size,
	int output_num,
	int pad,
	int stride) {

    int output_h = (height + 2 * pad - kernel_size) / stride + 1;
    int output_w = (width + 2 * pad - kernel_size) / stride + 1;

    int row_padding = 16 - (channels*kernel_size*kernel_size) % 16;

    int row_size = channels*kernel_size*kernel_size + row_padding;
    int col_size = output_h*output_w;

    int m = output_num;
    int n = col_size;
    int k = row_size;


	SharedArray<T> &weight_buffer = _gp_array[0];
	SharedArray<T> &input_buffer = _gp_array[1];
	SharedArray<T> &output_buffer = _gp_array[2];

	LoadWeightIntoGpu(weight_buffer,weight,m*k);
	LoadInputIntoGpu(input_buffer,input,height,width,channels,kernel_size,pad,stride);
	GemmKernel(&weight_buffer,&input_buffer,&output_buffer,m,n,k);
	getOutputFromGpu(output_buffer,output,m*n*16);
}


template<typename T>
void GManager<T>::Init_Gpu_Memory() {
	_gp_array[0].alloc(20*(1<<20));
	_gp_array[1].alloc(20*(1<<20));
	_gp_array[2].alloc(20*(1<<20));
}



template<typename T>
GManager<T>::GManager(int gpu_num) {
	Init_Gpu_Memory();
	Init_Kernel(gpu_num);
}


template<typename T>
void GManager<T>::LoadWeightIntoGpu(
	SharedArray<T> &_shared_array_buffer,
	const T *input_data_buffer,
	int data_size) {
	for(int i=0;i<data_size;i++) {
		_shared_array_buffer[i] = input_data_buffer[i];
	}
}


template<typename T>
T *GManager<T>::TransWeight2GpuFormat(T *original_data,int channels,int kernel_size,int output_num) {
  int size = channels*kernel_size*kernel_size;
  int padding = 16 - size % 16;

  T *formated_data = new T[(size+padding)*output_num];

  int a_pos = 0;
  int b_pos = 0;
  for(int i=0;i<output_num;i++) {
      for(int j=0;j<size;j++) {
          formated_data[a_pos++] = original_data[b_pos++];
      }

      for(int j=0;j<padding;j++) {
          formated_data[a_pos++] = 0;
      }
  }
}


template<typename T>
void GManager<T>::LoadInputIntoGpu(
    SharedArray<T> &_shared_array_buffer,
    const T *input_data_buffer,
    int input_height,
    int input_width,
    int input_channel,
    int kernel_size,
    int pad,
    int stride) {

    SharedArray<T> &output_data=_shared_array_buffer;

    const int row_padding = 16 - (input_channel*kernel_size*kernel_size) % 16;

    const T *input_data;

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
                            output_data[array_pos++] = 0.0;
                        }            

                    }
                }

            }

            for(int i=0;i<row_padding;i++) {
                 output_data[array_pos++] = 0.0;
            }

        }
    }    

}


float *get_weight(int output_num,int channels,int kernel_size) {
	float *weight = new float[output_num*channels*kernel_size*kernel_size];
	for(int i=0;i<output_num;i++) {
		for(int j=0;j<channels;j++) {
			for(int k=0;k<kernel_size*kernel_size;k++) {
				*weight++ = rand()/(RAND_MAX+1.0);
			}
		}
	}
	return weight;
}


float *get_input(int height,int width,int channels) {
	float *input = new float[height*width*channels];
	for(int i=0;i<channels;i++) {
		for(int j=0;j<height;j++) {
			for(int k=0;k<width;k++) {
				*input++ = rand()/(RAND_MAX+1.0);
			}
		}
	}
	return input;
}

int main() {

    int output_num = 2;

    int channels = 2;
    int height = 4;
    int width = 4;
    int pad = 2;
    int stride = 2;
    int kernel_size = 3;

    int output_h = (height + 2 * pad - kernel_size) / stride + 1;
    int output_w = (width + 2 * pad - kernel_size) / stride + 1;

	GManager<float> gm(1);
	// float *weight = get_weight(output_num,channels,kernel_size);
	// float *input = get_input(height,width,channels);
	// float *output = new float[output_h*output_w];

	// float *gpu_format_weight = gm.TransWeight2GpuFormat(weight,channels,kernel_size,output_num);

	// gm.gpu_conv(gpu_format_weight,input,output,height,width,channels,kernel_size,output_num,pad,stride); 

}