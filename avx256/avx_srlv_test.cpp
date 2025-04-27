#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

int main() {
    // 初始化输入向量，包含 4 个 64 位整数
    __m256i inputVec = _mm256_setr_epi64x(
        0x123456789ABCDEF0ULL, 0x23456789ABCDEF0ULL, 0x3456789ABCDEF0ULL, 0x456789ABCDEF0123ULL
    );

    // 初始化移位计数向量
    __m256i shiftCount = _mm256_setr_epi64x(4, 4, 12, 16);

    // 进行逻辑右移操作
    // __m256i outputVec = _mm256_srlv_epi64(inputVec, shiftCount);
    __m256i outputVec = _mm256_srli_epi64(inputVec, 4);

    // 将结果存储到数组中以便打印
    uint64_t res[4];
    _mm256_storeu_si256((__m256i*)res, outputVec);

    // // 打印原始向量
    // printf("Original vector: ");
    // for (int i = 0; i < 4; i++) {
    //     printf("%016llx ", (unsigned long long)_mm256_extract_epi64(inputVec, i));
    // }
    // printf("\n");

    // // 打印移位计数向量
    // printf("Shift counts: ");
    // for (int i = 0; i < 4; i++) {
    //     printf("%lld ", (long long)_mm256_extract_epi64(shiftCount, i));
    // }
    // printf("\n");

    // 打印结果向量
    printf("Result vector: ");
    for (int i = 0; i < 4; i++) {
        printf("%016llx ", (unsigned long long)res[i]);
    }
    printf("\n");

    return 0;
}