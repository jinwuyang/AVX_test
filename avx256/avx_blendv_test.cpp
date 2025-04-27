#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

int main() {
    // 初始化两个输入向量
    __m256i vec1 = _mm256_set_epi32(1, 2000, 3, 4, 5, 6, 7, 8);
    __m256i vec2 = _mm256_set_epi32(8, 7000, 6, 5, 4, 3, 2, 1);

    // 初始化掩码向量
    // 掩码的每一位对应一个字节，这里我们使用一个简单的掩码模式：0x0F0F0F0F0F0F0F0F
    // 这个掩码表示奇数位置的字节为 0xFF（真），偶数位置的字节为 0x00（假）
    __m256i mask = _mm256_set_epi32(
        0, 33023, 0, 0, 0, -1, 0, 0
    );

    // 使用 _mm256_blendv_epi8 进行混合
    __m256i result = _mm256_blendv_epi8(vec1, vec2, mask);

    // 将结果存储到数组中以便打印
    int res[8];
    _mm256_storeu_si256((__m256i*)res, result);

    // 打印结果

    printf("mask: ");
    for (int i = 0; i < 8; i++) {
        printf("%02X ", ((unsigned char*)&mask)[i]);
    }
    printf("\n");

    printf("result: ");
    for (int i = 0; i < 8; i++) {
        printf("%d ", res[i]);
    }
    printf("\n");

    return 0;
}