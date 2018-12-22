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

class GManager {
public:
	GManager();


private:
	void Init_Gpu_Memory();

	SharedArray<float> _gp_array[3];
};


void GManager::Init_Gpu_Memory() {
	_gp_array[0].alloc(21*(1<<20));
	_gp_array[1].alloc(21*(1<<20));
	_gp_array[2].alloc(22*(1<<20));
}

GManager::GManager() {
	Init_Gpu_Memory();
}


int main() {
	GManager gm;
}
