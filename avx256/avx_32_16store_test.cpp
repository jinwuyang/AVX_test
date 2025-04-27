#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

int main() {
    // 初始化输入向量，包含 8 个 32 位整数
    __m256i inputVec = _mm256_setr_epi32(
        0x12345678, 0x23456789, 0x3456789A, 0x456789AB,
        0x56789ABC, 0x6789ABCD, 0x789ABCDE, 0x89ABCDEF
    );

    // 创建一个掩码，用于提取低 16 位
    __m256i mask = _mm256_set1_epi32(0x0000FFFF);

    // 使用按位与操作提取低 16 位
    __m256i low16Vec = _mm256_and_si256(inputVec, mask);

    // 将 32 位整数转换为 16 位无符号整数
    __m256i outputVec = _mm256_packus_epi32(low16Vec, _mm256_setzero_si256());

    // 将结果存储到数组中以便打印
    uint16_t res[16];
    _mm256_storeu_si256((__m256i*)res, outputVec);

    // 打印原始向量
    // printf("Original vector: ");
    // for (int i = 0; i < 8; i++) {
    //     printf("%08X ", _mm256_extract_epi32(inputVec, i));
    // }
    // printf("\n");

    // 打印结果向量
    printf("Output vector: ");
    for (int i = 0; i < 8; i++) {
        printf("%04X ", res[i]);
    }
    printf("\n");

    return 0;
}