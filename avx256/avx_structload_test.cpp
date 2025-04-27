#include <immintrin.h>
#include <stdio.h>

struct MyStruct {
    int a, b, c, d;
};

int main() {
    // 创建一个结构体数组
    MyStruct arr[8] = { {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16},
                        {17, 18, 19, 20}, {21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32} };

    // 定义用于存储结构体各字段的向量
    __m256i a_vec, b_vec, c_vec, d_vec;

    // 批量加载结构体中的 a、b、c、d 字段
    // 注意：这里假设结构体数组在内存中连续存储，且每个字段对齐到 32 字节边界。
    a_vec = _mm256_loadu_si256((__m256i*)(&arr[0].a));
    b_vec = _mm256_loadu_si256((__m256i*)(&arr[0].b));
    c_vec = _mm256_loadu_si256((__m256i*)(&arr[0].c));
    d_vec = _mm256_loadu_si256((__m256i*)(&arr[0].d));

    // 打印加载的向量值
    int res_a[8], res_b[8], res_c[8], res_d[8];
    _mm256_storeu_si256((__m256i*)res_a, a_vec);
    _mm256_storeu_si256((__m256i*)res_b, b_vec);
    _mm256_storeu_si256((__m256i*)res_c, c_vec);
    _mm256_storeu_si256((__m256i*)res_d, d_vec);

    for (int i = 0; i < 8; i++) {
        printf("arr[%d].a = %d, arr[%d].b = %d, arr[%d].c = %d, arr[%d].d = %d\n", i, res_a[i], i, res_b[i], i, res_c[i], i, res_d[i]);
    }

    return 0;
}