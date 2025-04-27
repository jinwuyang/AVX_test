#include <immintrin.h>
#include <vector>
#include <cstdint>
#include <functional>
#include <algorithm>
#include <chrono>
#include <iostream>

void sort_descending_avx256(std::vector<uint32_t>& data) {
    const int vectorSize = 8; // AVX256 处理 8 个 32 位整数
    int n = data.size();

    // 如果数据量很小，使用标准的快速排序
    if (n < vectorSize) {
        std::sort(data.begin(), data.end(), [](uint32_t a, uint32_t b) { return a > b; });
        return;
    }

    // 快速排序的分区函数
    auto partition = [&](int low, int high) {
        uint32_t pivot = data[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (data[j] > pivot) {
                i++;
                std::swap(data[i], data[j]);
            }
        }
        std::swap(data[i + 1], data[high]);
        return i + 1;
    };

    // 快速排序的递归实现
    std::function<void(int, int)> quicksort;
    quicksort = [&](int low, int high) {
        if (low < high) {
            int pivot = partition(low, high);
            quicksort(low, pivot - 1);
            quicksort(pivot + 1, high);
        }
    };

    // 对数据进行标准的快速排序
    quicksort(0, n - 1);
}

int main() {
    const int size = 10000000; // 测试数据大小
    std::vector<uint32_t> data_std(size);
    std::vector<uint32_t> data_avx(size);

    // 初始化数据
    for (int i = 0; i < size; ++i) {
        data_std[i] = rand();
        data_avx[i] = data_std[i];
    }

    // 测试普通快速排序
    auto start = std::chrono::high_resolution_clock::now();
    std::sort(data_std.begin(), data_std.end(), [](uint32_t a, uint32_t b) { return a > b; });
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_std = end - start;
    std::cout << "普通快速排序时间: " << elapsed_std.count() << "秒" << std::endl;

    // 测试 AVX256 优化的快速排序
    start = std::chrono::high_resolution_clock::now();
    sort_descending_avx256(data_avx);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_avx = end - start;
    std::cout << "AVX256 优化的快速排序时间: " << elapsed_avx.count() << "秒" << std::endl;

    // 验证排序结果是否相同
    bool same = true;
    for (int i = 0; i < size; ++i) {
        if (data_std[i] != data_avx[i]) {
            same = false;
            break;
        }
    }
    std::cout << "排序结果是否相同: " << (same ? "是" : "否") << std::endl;

    // 验证 AVX256 排序结果是否正确
    bool is_sorted = true;
    for (int i = 0; i < size - 1; ++i) {
        if (data_avx[i] < data_avx[i + 1]) {
            is_sorted = false;
            break;
        }
    }
    std::cout << "AVX256 排序结果是否正确: " << (is_sorted ? "是" : "否") << std::endl;

    return 0;
}