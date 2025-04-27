#include <immintrin.h>
#include <stdio.h>
#include <string.h>

// 确保内存对齐（32字节对齐）
alignas(32) static float input[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
alignas(32) static float output[8];
alignas(32) static float expected[8] = {0.0f, 1.0f, 3.0f, 6.0f, 10.0f, 15.0f, 21.0f, 28.0f};

#include <immintrin.h>

// 修正版右移1个元素（跨通道优化）
static inline __m256 shift_r1_exclusive(__m256 v) {
    // 跨通道右移1元素：将高128位的后3元素与低128位的前1元素组合
    __m256 v_hi = _mm256_permute2f128_ps(v, v, 0x81); // [高128位 | 低128位] → [低128位的高半部分 | 高128位的低半部分]
    __m256 shifted = _mm256_permute_ps(v_hi, _MM_SHUFFLE(2, 1, 0, 3)); // 通道内右移1元素：[A,B,C,D] → [D,A,B,C]
    return _mm256_blend_ps(_mm256_setzero_ps(), shifted, 0xFE); // 首位置零
}

// 修正后的快速独占前缀和
__m256 exclusive_prefix_sum_avx_fixed(__m256 v) {
    // 步骤1：右移1位并累加（覆盖相邻元素）
    __m256 s1 = shift_r1_exclusive(v);
    __m256 sum1 = _mm256_add_ps(v, s1); // [a, a+b, b+c, c+d, ...]

    // 步骤2：右移2位并累加（覆盖前两元素）
    __m256 s2 = _mm256_permute_ps(sum1, _MM_SHUFFLE(1, 0, 3, 2)); // 每128位内交换前两和后两元素
    s2 = _mm256_permute2f128_ps(s2, s2, 0x81); // 高低128位交换
    __m256 sum2 = _mm256_add_ps(sum1, s2); // [a, a+b, a+b+c, a+b+c+d, ...]

    // 步骤3：右移4位并累加（覆盖前四元素）
    __m256 s3 = _mm256_permute2f128_ps(sum2, sum2, 0x01); // 交换高低128位
    s3 = _mm256_permute_ps(s3, _MM_SHUFFLE(2, 1, 0, 3)); // 右移1元素
    s3 = _mm256_blend_ps(_mm256_setzero_ps(), s3, 0xF0); // 仅保留低4位
    __m256 sum3 = _mm256_add_ps(sum2, s3);

    // 转换为独占前缀和：右移1位并置零首位
    return shift_r1_exclusive(sum3);
}

// 打印数组内容
void print_array(const char* name, const float* arr) {
    printf("%-8s: [", name);
    for (int i = 0; i < 8; i++) {
        printf("%5.1f", arr[i]);
        if (i < 7) printf(", ");
    }
    printf("]\n");
}

// 主测试函数
bool test_exclusive_prefix_sum() {
    // 加载输入数据到AVX寄存器
    __m256 vec = _mm256_load_ps(input);
    
    // 调用AVX优化函数
    __m256 result = exclusive_prefix_sum_avx_fixed(vec);
    
    // 将结果存回内存
    _mm256_store_ps(output, result);
    
    // 检查结果是否正确
    bool success = true;
    for (int i = 0; i < 8; i++) {
        if (output[i] != expected[i]) {
            success = false;
            break;
        }
    }
    
    // 打印调试信息
    print_array("Input", input);
    print_array("Output", output);
    print_array("Expected", expected);
    
    return success;
}

int main() {
    bool passed = test_exclusive_prefix_sum();
    printf("\nTest %s!\n", passed ? "Passed" : "Failed");
    return passed ? 0 : 1;
}