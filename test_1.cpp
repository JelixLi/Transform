#include <iostream>

using namespace std;

const int Max_GPU_Memory = 10;


void LoadWeightIntoGpu(
	float *_shared_array_buffer,
	float *input_data_buffer,
	int group_size,
	int data_size) {
	float *input_data_buffer_ptr = input_data_buffer;
	int pos = 0;
	for(int j=0;j<group_size;j++) {
		for(int i=0;i<data_size;i++) {
			_shared_array_buffer[pos++] = *input_data_buffer_ptr++;
		}
	}
}

void Init_Matrix(float *A,int m,int n) {
	for(int i=0;i<m;i++) {
		for(int j=0;j<n;j++) {
			A[i][j] = rand() % 10;
		}
	}
}

int main() {
	int m = 10;
	int k = 10;
	int n = 10;
	float *A = new float[m*k*16];
	float *B = new float[k*n*16];
	float *C = new float[m*n];
	float *D = new float[m*n];

	Init_Matrix(A,m,k);
	Init_Matrix(B,k,n);


}