#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

int main() {
    // 初始化输入向量，包含 8 个 32 位整数
    __m256i inputVec = _mm256_setr_epi32(
        0x12345678, 0x23456789, 0x3456789A, 0x456789AB,
        0x56789ABC, 0x6789ABCD, 0x789ABCDE, 0x89ABCDEF
    );

    // 将每个 128 位通道左移 8 个字节（即 64 位）
    __m256i result = _mm256_slli_si256(inputVec, 8);

    // 将结果存储到数组中以便打印
    uint32_t res[8];
    _mm256_storeu_si256((__m256i*)res, result);

    // // 打印原始向量
    // printf("Original vector: ");
    // for (int i = 0; i < 8; i++) {
    //     printf("%08X ", _mm256_extract_epi32(inputVec, i));
    // }
    // printf("\n");

    // 打印移位后的向量
    printf("Shifted vector: ");
    for (int i = 0; i < 8; i++) {
        printf("%08X ", res[i]);
    }
    printf("\n");

    return 0;
}