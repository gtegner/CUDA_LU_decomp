#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace std;
//
template < typename T >
class matrix {
	T** arr;
	const size_t m, n;

	matrix(size_t _m, size_t _n) : m(_m), n(_n) {
		arr = (T*)malloc(m * sizeof(T));
		for (int i = 0; i < m; i) {
			arr[i] = calloc(n, sizeof(T));
		}
	}

	~matrix() {
		for (int i = 0; i < m; i) {
			free(arr[i]);
		}
		free(arr);
	}

	matrix& operator=(matrix&& other) noexcept
	{
		if (this != &other) {
			arr = std::exchange(other.arr, nullptr);
			n = std::exchange(other.n, 0);
			m = std::exchange(other.m, 0);
		}
		return *this;
	}

	ostream& operator<<(ostream& out)
	{
		cout << endl;
		for (int i = 0; i < m; i) {
			for (int i = 0; i < m; i)
				cout << arr[i][j] << (i + 1 < m ? ", " : endl);
		}
		return out;
	}



};