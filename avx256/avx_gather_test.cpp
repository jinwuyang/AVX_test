#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

int main() {
    constexpr int SIMD_LANES = 8;
    // 定义数据数组
    uint8_t data[32] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 10, 20, 30, 40, 50, 60, 70};
    // uint8_t data[32] = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
    // 定义索引数组
    // int indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    __m256i indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i offset = _mm256_setr_epi32(0, 1*256, 2*256, 3*256, 4*256, 5*256, 6*256, 7*256);
    uint32_t hist_matrix[SIMD_LANES][256] = {{0}};
    uint32_t global_hist[256] = {0};

    // 初始化结果向量
    // __m256i result = _mm256_setzero_si256();
    // __m256i result0 = _mm256_setzero_si256();
    // __m256i result1 = _mm256_setzero_si256();
    // __m256i result2 = _mm256_setzero_si256();
    // __m256i result3 = _mm256_setzero_si256();

    // 使用_mm256_i32gather_epi32收集数据
    // __m256i index_vector = _mm256_loadu_si256((__m256i*)indices);
    __m256i result = _mm256_i32gather_epi32(data, indices, 4);
    // __m256i result0 = _mm256_and_si256(result, _mm256_set1_epi32(255));
    // // result1 = _mm256_and_si256(result, _mm256_set1_epi32(65280));
    // __m256i result1 = _mm256_srli_epi32(_mm256_and_si256(result, _mm256_set1_epi32(65280)), 8);
    // // result2 = _mm256_and_si256(result, _mm256_set1_epi32(16711680));
    // __m256i result2 = _mm256_srli_epi32(_mm256_and_si256(result, _mm256_set1_epi32(16711680)), 16);
    // // result3 = _mm256_and_si256(result, _mm256_set1_epi32(4278190080));
    // __m256i result3 = _mm256_srli_epi32(_mm256_and_si256(result, _mm256_set1_epi32(4278190080)), 24);

    // result0 = _mm256_add_epi32(result0, offset);
    // result1 = _mm256_add_epi32(result1, offset);
    // result2 = _mm256_add_epi32(result2, offset);
    // result3 = _mm256_add_epi32(result3, offset);

    __m256i result0 = _mm256_add_epi32(_mm256_and_si256(result, _mm256_set1_epi32(255)), offset);
    __m256i result1 = _mm256_add_epi32(_mm256_srli_epi32(_mm256_and_si256(result, _mm256_set1_epi32(65280)), 8), offset);
    __m256i result2 = _mm256_add_epi32(_mm256_srli_epi32(_mm256_and_si256(result, _mm256_set1_epi32(16711680)), 16), offset);
    __m256i result3 = _mm256_add_epi32(_mm256_srli_epi32(_mm256_and_si256(result, _mm256_set1_epi32(4278190080)), 24), offset);

    // __m256i index0 = _mm256_i32gather_epi32(global_hist, result0, 4);
    // index0 = _mm256_add_epi32(index0, _mm256_set1_epi32(1));
    // _mm256_i32scatter_epi32(global_hist, result0, index0, 4);

    _mm256_i32scatter_epi32(hist_matrix, result0, _mm256_add_epi32(_mm256_i32gather_epi32(hist_matrix, result0, 4), _mm256_set1_epi32(1)), 4);

    // __m256i index1 = _mm256_i32gather_epi32(global_hist, result1, 4);
    // index1 = _mm256_add_epi32(index1, _mm256_set1_epi32(1));
    // _mm256_i32scatter_epi32(global_hist, result1, index1, 4);

    _mm256_i32scatter_epi32(hist_matrix, result1, _mm256_add_epi32(_mm256_i32gather_epi32(hist_matrix, result1, 4), _mm256_set1_epi32(1)), 4);

    // __m256i index2 = _mm256_i32gather_epi32(global_hist, result2, 4);
    // index2 = _mm256_add_epi32(index2, _mm256_set1_epi32(1));
    // _mm256_i32scatter_epi32(global_hist, result2, index2, 4);

    _mm256_i32scatter_epi32(hist_matrix, result2, _mm256_add_epi32(_mm256_i32gather_epi32(hist_matrix, result2, 4), _mm256_set1_epi32(1)), 4);

    // __m256i index3 = _mm256_i32gather_epi32(global_hist, result3, 4);
    // index3 = _mm256_add_epi32(index3, _mm256_set1_epi32(1));
    // _mm256_i32scatter_epi32(global_hist, result3, index3, 4);

    _mm256_i32scatter_epi32(hist_matrix, result3, _mm256_add_epi32(_mm256_i32gather_epi32(hist_matrix, result3, 4), _mm256_set1_epi32(1)), 4);

    // 打印结果
    // int res0[8];
    // _mm256_storeu_si256((__m256i*)res0, result0);
    // for (int lane = 0; lane < SIMD_LANES; lane++) {
    //     hist_matrix[lane][_mm256_extract_epi32(result0, lane)]++;
    // }
    // printf("\n");

    // int res1[8];
    // _mm256_storeu_si256((__m256i*)res1, result1);
    // for (int lane = 0; lane < SIMD_LANES; lane++) {
    //     hist_matrix[lane][_mm256_extract_epi32(result1, lane)]++;
    // }
    // printf("\n");

    // int res2[8];
    // _mm256_storeu_si256((__m256i*)res2, result2);
    // for (int lane = 0; lane < SIMD_LANES; lane++) {
    //     hist_matrix[lane][_mm256_extract_epi32(result2, lane)]++;
    // }
    // printf("\n");

    // int res3[8];
    // _mm256_storeu_si256((__m256i*)res3, result3);
    // for (int lane = 0; lane < SIMD_LANES; lane++) {
    //     hist_matrix[lane][_mm256_extract_epi32(result3, lane)]++;
    // }
    // printf("\n");
    // hist_matrix[0][_mm256_extract_epi32(result0, 0)]++;
    // hist_matrix[0][_mm256_extract_epi32(result1, 0)]++;
    // hist_matrix[0][_mm256_extract_epi32(result2, 0)]++;
    // hist_matrix[0][_mm256_extract_epi32(result3, 0)]++;
    // hist_matrix[1][_mm256_extract_epi32(result0, 1)]++;
    // hist_matrix[1][_mm256_extract_epi32(result1, 1)]++;
    // hist_matrix[1][_mm256_extract_epi32(result2, 1)]++;
    // hist_matrix[1][_mm256_extract_epi32(result3, 1)]++;
    // hist_matrix[2][_mm256_extract_epi32(result0, 2)]++;
    // hist_matrix[2][_mm256_extract_epi32(result1, 2)]++;
    // hist_matrix[2][_mm256_extract_epi32(result2, 2)]++;
    // hist_matrix[2][_mm256_extract_epi32(result3, 2)]++;
    // hist_matrix[3][_mm256_extract_epi32(result0, 3)]++;
    // hist_matrix[3][_mm256_extract_epi32(result1, 3)]++;
    // hist_matrix[3][_mm256_extract_epi32(result2, 3)]++;
    // hist_matrix[3][_mm256_extract_epi32(result3, 3)]++;
    // hist_matrix[4][_mm256_extract_epi32(result0, 4)]++;
    // hist_matrix[4][_mm256_extract_epi32(result1, 4)]++;
    // hist_matrix[4][_mm256_extract_epi32(result2, 4)]++;
    // hist_matrix[4][_mm256_extract_epi32(result3, 4)]++;
    // hist_matrix[5][_mm256_extract_epi32(result0, 5)]++;
    // hist_matrix[5][_mm256_extract_epi32(result1, 5)]++;
    // hist_matrix[5][_mm256_extract_epi32(result2, 5)]++;
    // hist_matrix[5][_mm256_extract_epi32(result3, 5)]++;
    // hist_matrix[6][_mm256_extract_epi32(result0, 6)]++;
    // hist_matrix[6][_mm256_extract_epi32(result1, 6)]++;
    // hist_matrix[6][_mm256_extract_epi32(result2, 6)]++;
    // hist_matrix[6][_mm256_extract_epi32(result3, 6)]++;
    // hist_matrix[7][_mm256_extract_epi32(result0, 7)]++;
    // hist_matrix[7][_mm256_extract_epi32(result1, 7)]++;
    // hist_matrix[7][_mm256_extract_epi32(result2, 7)]++;
    // hist_matrix[7][_mm256_extract_epi32(result3, 7)]++;

    // hist_matrix[0][_mm256_extract_epi32(result0, 0)]++;
    // hist_matrix[1][_mm256_extract_epi32(result0, 1)]++;
    // hist_matrix[2][_mm256_extract_epi32(result0, 2)]++;
    // hist_matrix[3][_mm256_extract_epi32(result0, 3)]++;
    // hist_matrix[4][_mm256_extract_epi32(result0, 4)]++;
    // hist_matrix[5][_mm256_extract_epi32(result0, 5)]++;
    // hist_matrix[6][_mm256_extract_epi32(result0, 6)]++;
    // hist_matrix[7][_mm256_extract_epi32(result0, 7)]++;
    // hist_matrix[0][_mm256_extract_epi32(result1, 0)]++;
    // hist_matrix[1][_mm256_extract_epi32(result1, 1)]++;
    // hist_matrix[2][_mm256_extract_epi32(result1, 2)]++;
    // hist_matrix[3][_mm256_extract_epi32(result1, 3)]++;
    // hist_matrix[4][_mm256_extract_epi32(result1, 4)]++;
    // hist_matrix[5][_mm256_extract_epi32(result1, 5)]++; 
    // hist_matrix[6][_mm256_extract_epi32(result1, 6)]++;
    // hist_matrix[7][_mm256_extract_epi32(result1, 7)]++;
    // hist_matrix[0][_mm256_extract_epi32(result2, 0)]++;
    // hist_matrix[1][_mm256_extract_epi32(result2, 1)]++;
    // hist_matrix[2][_mm256_extract_epi32(result2, 2)]++;
    // hist_matrix[3][_mm256_extract_epi32(result2, 3)]++;
    // hist_matrix[4][_mm256_extract_epi32(result2, 4)]++;
    // hist_matrix[5][_mm256_extract_epi32(result2, 5)]++;
    // hist_matrix[6][_mm256_extract_epi32(result2, 6)]++;
    // hist_matrix[7][_mm256_extract_epi32(result2, 7)]++;
    // hist_matrix[0][_mm256_extract_epi32(result3, 0)]++;
    // hist_matrix[1][_mm256_extract_epi32(result3, 1)]++;
    // hist_matrix[2][_mm256_extract_epi32(result3, 2)]++;
    // hist_matrix[3][_mm256_extract_epi32(result3, 3)]++;
    // hist_matrix[4][_mm256_extract_epi32(result3, 4)]++;
    // hist_matrix[5][_mm256_extract_epi32(result3, 5)]++;
    // hist_matrix[6][_mm256_extract_epi32(result3, 6)]++;
    // hist_matrix[7][_mm256_extract_epi32(result3, 7)]++;


    for (int lane = 0; lane < SIMD_LANES; lane++) {
        for (int i = 0; i < 256; i++) {
            global_hist[i] += hist_matrix[lane][i];
        }
    }

    // 打印结果
    for (int i = 0; i < 256; i++) {
        printf("%d ", global_hist[i]);
    }
    printf("\n");

    return 0;
}