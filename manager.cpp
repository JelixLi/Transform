#include <stdlib.h> 
#include <time.h> 
#include <stdio.h>
#include <iostream>
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

    Int iNNC = 16;
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
           For(Int s=0,s<k,s=s+iNNC)
              gather(p+iNNC);
              gather(q+iNNC);
              receive(x);
              receive(y);
              sum = sum + x*y;
              p=p+iNNC;
              q=q+iNNC;
           End
           receive(x);
           receive(y);
           store(sum,C + ind + ((r*n+c)<<4));
      End 
    End   
}



template<typename T>
class GManager {
public:
	GManager();

  void LoadDataIntoGpu(
    SharedArray<T> &_shared_array_buffer,
    T *input_data_buffer,
    int group_size,
    int data_size);

  void TransInput2GpuFormat(
    T *input_data_buffer,
    const T *input_data,
    int input_height,
    int input_width,
    int input_channel,
    int kernel_size,
    int pad,
    int stride);

  void GetOutputFromGpu(
    SharedArray<T> &_shared_array_buffer,
    T *output_data_buffer,
    int offset,
    int step_size,
    int row_size,
    int col_size);

  void gpu_conv(
    T *weight,
    T *input,
    T *output,
    int height,
    int width,
    int channels,
    int kernel_size,
    int output_num,
    int pad,
    int stride,
    auto& GemmKernel);

private:
	void Init_Gpu_Memory();


	inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
	    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
	}


	SharedArray<T> _gp_array[3];

  int Max_GPU_Memory; // float(4 bytes)

};


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
  int stride,
  auto& GemmKernel) {

    int output_h = (height + 2 * pad - kernel_size) / stride + 1;
    int output_w = (width + 2 * pad - kernel_size) / stride + 1;

    int row_padding = 16 - (channels*kernel_size*kernel_size) % 16;

    int row_size = channels*kernel_size*kernel_size + row_padding;
    int col_size = output_h*output_w;

    int m = output_num;
    int n = col_size;
    int k = row_size;

    SharedArray<T>& weight_buffer = _gp_array[0];
    SharedArray<T>& input_buffer = _gp_array[1];
    SharedArray<T>& output_buffer = _gp_array[2];

    int Gpu_Memory_Basic_Block = 100;

    int m_group = m / Gpu_Memory_Basic_Block;
    int _m_group = m % Gpu_Memory_Basic_Block;

    int n_group = n / Gpu_Memory_Basic_Block;
    int _n_group = n % Gpu_Memory_Basic_Block;

    for(int i=0;i<m_group+1;i++) {
      int weight_offset = i*k*Gpu_Memory_Basic_Block;
      int weight_group_size = ((i==m_group||m_group==0)?_m_group:Gpu_Memory_Basic_Block);

      LoadDataIntoGpu(
        weight_buffer,
        weight+weight_offset,
        weight_group_size,
        k);

      for(int j=0;j<n_group+1;j++) {
        int input_offset = j*k*Gpu_Memory_Basic_Block;
        int input_group_size = ((j==n_group||n_group==0)?_n_group:Gpu_Memory_Basic_Block);

        LoadDataIntoGpu(
          input_buffer,
          input+input_offset,
          input_group_size,
          k);

        GemmKernel(
          &weight_buffer,
          &input_buffer,
          &output_buffer,
          weight_group_size,
          input_group_size,
          k);

        GetOutputFromGpu(
          output_buffer,
          output,
          i*Gpu_Memory_Basic_Block*n+j*Gpu_Memory_Basic_Block,
          n,
          weight_group_size,
          input_group_size);
    }
  }
}

template<typename T>
void GManager<T>::GetOutputFromGpu(
    SharedArray<T> &_shared_array_buffer,
    T *output_data_buffer,
    int offset,
    int step_size,
    int row_size,
    int col_size) {

    int pos = 0;
    T *output = output_data_buffer + offset;
    int row_offset = offset / step_size;
    int col_offset = offset % step_size;

    for(int i=0;i<row_size;i++) {
      for(int j=0;j<col_size;j++) {
        float sum = 0;
        for(int k=0;k<16;k+=4) {
            sum += _shared_array_buffer[pos+k];
            sum += _shared_array_buffer[pos+k+1];
            sum += _shared_array_buffer[pos+k+2];
            sum += _shared_array_buffer[pos+k+3];
        }
        pos += 16;
        output_data_buffer[(i+row_offset)*step_size+j+col_offset] = sum;
      }

    }

}

template<typename T>
void GManager<T>::TransInput2GpuFormat(
  T *output_data,
  const T *input_data_buffer,
  int input_height,
  int input_width,
  int input_channel,
  int kernel_size,
  int pad,
  int stride) {

    const T *input_data;

    for(int row=-pad;row<input_height+pad-kernel_size+1;row+=stride) {
        for(int col=-pad;col<input_width+pad-kernel_size+1;col+=stride) {

            for(int chan=0;chan<input_channel;chan++) {

                input_data = input_data_buffer + chan*input_height*input_width;

                for(int kernel_row=0;kernel_row<kernel_size;kernel_row++) {
                    for(int kernel_col=0;kernel_col<kernel_size;kernel_col++) {

                        int new_row=row+kernel_row;
                        int new_col=col+kernel_col;

                        if(is_a_ge_zero_and_a_lt_b(new_row,input_height)&&is_a_ge_zero_and_a_lt_b(new_col,input_width)) {
                            *output_data++ = input_data[new_row*input_width+new_col];
                        } else {
                            *output_data++ = 0.0;
                        }            

                    }
                }

            }
        }
    }    

}


template<typename T>
void GManager<T>::LoadDataIntoGpu(
  SharedArray<T> &_shared_array_buffer,
  T *input_data_buffer,
  int group_size,
  int data_size)  {

  T *input_data_buffer_ptr = input_data_buffer;
  int pos = 0;
  int size = data_size / 16 * 16;
  int _size = data_size % 16;
  for(int i=0;i<group_size;i++) {
    for(int j=0;j<size;j++) {
      _shared_array_buffer[pos++] = *input_data_buffer_ptr++;
    }
    for(int j=0;j<_size;j++) {
      _shared_array_buffer[pos++] = 0.0;
    }
  }
 
}




template<typename T>
void GManager<T>::Init_Gpu_Memory() {
	_gp_array[0].alloc(Max_GPU_Memory/3);
	_gp_array[1].alloc(Max_GPU_Memory/3);
	_gp_array[2].alloc(Max_GPU_Memory/3);
}



template<typename T>
GManager<T>::GManager():Max_GPU_Memory(733409) {
	Init_Gpu_Memory();
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

    auto GemmKernel = compile(gpu_gemm);
    GemmKernel.setNumQPUs(12);

    int output_num = 64;

    int channels = 3;
    int height = 227;
    int width = 227;
    int pad = 0;
    int stride = 2;
    int kernel_size = 3;

    int output_h = (height + 2 * pad - kernel_size) / stride + 1; 
    int output_w = (width + 2 * pad - kernel_size) / stride + 1; 

    GManager<float> gm;
    float *weight = get_weight(output_num,channels,kernel_size);
    float *input = get_input(height,width,channels);
    float *output = new float[output_h*output_w*output_num];

    float *col_data = new float[kernel_size*kernel_size*channels*output_w*output_h];

    clock_t start=clock();
    gm.TransInput2GpuFormat(
      col_data,
      input,
      height,
      width,
      channels,
      kernel_size,
      pad,
      stride);


    gm.gpu_conv(
      weight,
      col_data,
      output,
      height,
      width,
      channels,
      kernel_size,
      output_num,
      pad,
      stride,
      GemmKernel);

    clock_t end=clock();
    printf("gpu_cost: %f\n",(end-start)/double(CLOCKS_PER_SEC)*1000);

}

