#include <stdlib.h>
#include "QPULib.h"

void gcd(Ptr<Int> p, Ptr<Int> q, Ptr<Int> r)
{
  Int a = *p;
  Int b = *q;
  While (any(a != b))
    Where (a > b)
      a = a-b;
    End
    Where (a < b)
      b = b-a;
    End
  End
  *r = a;
}

void Init() {
  // Construct kernel
  auto k = compile(gcd);

  // Allocate and initialise arrays shared between ARM and GPU
  SharedArray<int> a(733409/3), b(733409/3), r(733409/3);
  srand(0);
  for (int i = 0; i < 16; i++) {
    a[i] = 100 + (rand() % 100);
    b[i] = 100 + (rand() % 100);
  }

  // Invoke the kernel and display the result
  k(&a, &b, &r);
  for (int i = 0; i < 16; i++)
    printf("gcd(%i, %i) = %i\n", a[i], b[i], r[i]);
}

int main()
{  
	Init();
  return 0;
}

// template<class T>
// class GManager {
// public:
// 	GManager();

// 	TransInput2GpuFormat(
// 	    T *input_data_buffer,
// 	    const T *input_data,
// 	    int input_height,
// 	    int input_width,
// 	    int input_channel,
// 	    int kernel_size,
// 	    int pad,
// 	    int stride);
// };


// template<typename T>
// void GManager<T>::TransInput2GpuFormat (
//     T *input_data_buffer,
//     const T *input_data,
//     int input_height,
//     int input_width,
//     int input_channel,
//     int kernel_size,
//     int pad,
//     int stride) {

//     T *output_data=input_data_buffer;

//     const int row_padding = 16 - (input_channel*kernel_size*kernel_size) % 16;

//     const T *input_data;

//     for(int row=-pad;row<input_height+pad-kernel_size+1;row+=stride) {
//         for(int col=-pad;col<input_width+pad-kernel_size+1;col+=stride) {

//             for(int chan=0;chan<input_channel;chan++) {

//                 input_data = input_data + chan*input_height*input_width;

//                 for(int kernel_row=0;kernel_row<kernel_size;kernel_row++) {
//                     for(int kernel_col=0;kernel_col<kernel_size;kernel_col++) {

//                         int new_row=row+kernel_row;
//                         int new_col=col+kernel_col;

//                         if(is_a_ge_zero_and_a_lt_b(new_row,input_height)&&is_a_ge_zero_and_a_lt_b(new_col,input_width)) {
//                             *output_data++ = input_data[new_row*input_width+new_col];
//                         } else {
//                             *output_data++ = 0.0;
//                         }            

//                     }
//                 }

//             }

//             for(int i=0;i<row_padding;i++) {
//                  *output_data = 0.0;
//             }

//         }
//     }    

// }
