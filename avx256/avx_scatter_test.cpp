#include <chrono>
#include <iostream>

#include <immintrin.h>
#include <stdint.h>

void scalar_update_pdf(const uint32_t* tidSymbol, const uint32_t* qProb, uint32_t* pdf, int kNumSymbols) {
    #pragma unroll(8)
    for (int i = 0; i < kNumSymbols; ++i) {
        pdf[tidSymbol[i]] = qProb[i];
    }
}

void avx256_update_pdf(const uint32_t* tidSymbol, const uint32_t* qProb, uint32_t* pdf, int kNumSymbols) {
    const int kVectorSize = 8; // AVX256 可以处理 8 个 32 位整数

    for (int i = 0; i < kNumSymbols; i += kVectorSize) {
        __m256i tidSymbolVec = _mm256_loadu_si256((__m256i*)(tidSymbol + i));
        __m256i qProbVec = _mm256_loadu_si256((__m256i*)(qProb + i));
        _mm256_i32scatter_epi32(pdf, tidSymbolVec, qProbVec, sizeof(uint32_t));
        // pdf[_mm256_extract_epi32(tidSymbolVec, 7)] = _mm256_extract_epi32(qProbVec, 7);
        // pdf[_mm256_extract_epi32(tidSymbolVec, 6)] = _mm256_extract_epi32(qProbVec, 6);
        // pdf[_mm256_extract_epi32(tidSymbolVec, 5)] = _mm256_extract_epi32(qProbVec, 5);
        // pdf[_mm256_extract_epi32(tidSymbolVec, 4)] = _mm256_extract_epi32(qProbVec, 4);
        // pdf[_mm256_extract_epi32(tidSymbolVec, 3)] = _mm256_extract_epi32(qProbVec, 3);
        // pdf[_mm256_extract_epi32(tidSymbolVec, 2)] = _mm256_extract_epi32(qProbVec, 2);
        // pdf[_mm256_extract_epi32(tidSymbolVec, 1)] = _mm256_extract_epi32(qProbVec, 1);
        // pdf[_mm256_extract_epi32(tidSymbolVec, 0)] = _mm256_extract_epi32(qProbVec, 0);
    }
}

void update_pdf_with_temp_array(const uint32_t* tidSymbol, const uint32_t* qProb, uint32_t* pdf, int kNumSymbols) {
    int index[8];
    #pragma unroll(4)
    for (int i = 0; i < kNumSymbols; i += 8) {
        __m256i tidSymbolVec = _mm256_loadu_si256((__m256i*)(tidSymbol + i));
        _mm256_storeu_si256((__m256i*)index, tidSymbolVec);
        #pragma unroll(8)
        for (int j = 0; j < 8; ++j) {
            pdf[index[j]] = qProb[i + j];
        }
    }
}

void test_performance(const uint32_t* tidSymbol, const uint32_t* qProb, uint32_t* pdf1, uint32_t* pdf2, int kNumSymbols) {
    // 测试标量化的代码
    auto start = std::chrono::high_resolution_clock::now();
    scalar_update_pdf(tidSymbol, qProb, pdf1, kNumSymbols);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Scalar time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

    // 测试 AVX256 化的代码
    start = std::chrono::high_resolution_clock::now();
    avx256_update_pdf(tidSymbol, qProb, pdf1, kNumSymbols);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "AVX256 time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

    // 测试使用临时数组的代码
    start = std::chrono::high_resolution_clock::now();
    update_pdf_with_temp_array(tidSymbol, qProb, pdf2, kNumSymbols);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Temp array time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

    // 验证结果是否一致
    bool same = true;
    for (int i = 0; i < kNumSymbols; ++i) {
        if (pdf1[i] != pdf2[i]) {
            same = false;
            break;
        }
    }
    std::cout << "Results are " << (same ? "same" : "different") << std::endl;
}

int main() {
    const int kNumSymbols = 1024;
    uint32_t tidSymbol[kNumSymbols];
    uint32_t qProb[kNumSymbols];
    uint32_t pdf1[kNumSymbols] = {0};
    uint32_t pdf2[kNumSymbols] = {0};

    // 初始化测试数据
    for (int i = 0; i < kNumSymbols; ++i) {
        tidSymbol[i] = i % kNumSymbols;
        qProb[i] = i + 1;
    }

    test_performance(tidSymbol, qProb, pdf1, pdf2, kNumSymbols);

    return 0;
}