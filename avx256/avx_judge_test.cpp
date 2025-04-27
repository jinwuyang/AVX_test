#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

int main() {
    const int kVectorSize = 8;
    __m256i qProbInt = _mm256_setr_epi32(0, 0, 1, 0, 0, 5, 0, 7);// 1 1 0 1 1 0 1 0
    __m256i countsInt = _mm256_setr_epi32(0, 2, 0, 4, 0, 6, 0, 8);// 0 1 0 1 0 1 0 1 
    __m256i one = _mm256_set1_epi32(1);
    __m256i zero = _mm256_setzero_si256();

    // 第一行代码
    __m256i result1 = _mm256_blendv_epi8(qProbInt, one, _mm256_and_si256(_mm256_cmpgt_epi32(countsInt, zero), _mm256_cmpeq_epi32(qProbInt, zero)));

    // 第二行代码
    __m256i result2 = _mm256_blendv_epi8(qProbInt, one, _mm256_andnot_si256(_mm256_cmpgt_epi32(qProbInt), countsInt));//0 1 0 0 0 0 0 0

    int qProbInt_out[kVectorSize];
    int result1_out[kVectorSize];
    int result2_out[kVectorSize];
    int temp[8];
    int temp2[8];
    _mm256_storeu_si256((__m256i*)temp, _mm256_andnot_si256(qProbInt, countsInt));
    _mm256_storeu_si256((__m256i*)temp2, _mm256_and_si256(_mm256_cmpgt_epi32(countsInt, zero), _mm256_cmpeq_epi32(qProbInt, zero)));
    for (int i = 0; i < kVectorSize; i++) {
        printf("%d ", temp[i]);
    }
    printf("\n");
    for(int i = 0; i < kVectorSize; i++) {
        printf("%d ", temp2[i]);
    }
    printf("\n");

    _mm256_storeu_si256((__m256i*)qProbInt_out, qProbInt);
    _mm256_storeu_si256((__m256i*)result1_out, result1);
    _mm256_storeu_si256((__m256i*)result2_out, result2);

    printf("Original qProbInt: ");
    for (int i = 0; i < kVectorSize; i++) {
        printf("%d ", qProbInt_out[i]);
    }
    printf("\n");

    printf("Result 1: ");
    for (int i = 0; i < kVectorSize; i++) {
        printf("%d ", result1_out[i]);
    }
    printf("\n");

    printf("Result 2: ");
    for (int i = 0; i < kVectorSize; i++) {
        printf("%d ", result2_out[i]);
    }
    printf("\n");

    return 0;
}