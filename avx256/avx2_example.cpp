#include <immintrin.h>
#include <iostream>

int main() {
    __m256i vec = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    __m256i state = _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1);
    __m256i result;

    __asm__ volatile (
        "vpmulld %[vec], %[state], %[result]"
        : [result] "=x"(result)
        : [vec] "x"(vec), [state] "x"(state)
    );

    // 将结果存储到数组中
    alignas(32) int32_t resultArray[8];
    _mm256_storeu_si256((__m256i*)resultArray, result);

    // 输出结果
    std::cout << "Result: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << resultArray[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
