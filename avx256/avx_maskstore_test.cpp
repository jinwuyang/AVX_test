#include <immintrin.h>
#include <stdio.h>

void example_mm256_maskstore_epi32() {
    // 定义一个数组作为目标内存
    int outWords[8] = {0};

    // 创建一个掩码寄存器，掩码值为 0b10101010，表示只写入偶数位置的元素
    __m256i mask = _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1);

    // 创建一个包含要写入数据的 AVX256 寄存器
    __m256i data = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);

    // 使用 _mm256_maskstore_epi32 将数据有条件地写入内存
    _mm256_maskstore_epi32(outWords, mask, data);

    // 打印结果
    for (int i = 0; i < 8; ++i) {
        printf("%d ", outWords[i]);
    }
}

int main() {
    example_mm256_maskstore_epi32();
    return 0;
}