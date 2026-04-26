#include <iostream>
#include <arm_neon.h>
#include <vector>
#include <chrono>

void vector_add(const float* A, const float* B, float* C, size_t n) {
    size_t i = 0;

    for (; i < n - 4; i+=4) {
        float32x4_t va = vld1q_f32(&A[i]);
        float32x4_t vb = vld1q_f32(&B[i]);

        float32x4_t vc = vaddq_f32(va, vb);

        vst1q_f32(&C[i], vc);
    }

    for (; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    size_t N = 100000;

    std::vector<float> A(N, 1.5f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N, 0.0f);

    auto start = std::chrono::steady_clock::now();
    vector_add(A.data(), B.data(), C.data(), N);
    auto end =  std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Computed " << N << " elements in " << duration.count() << " ms\n";
    std::cout << "Sample result (C[0]): " << C[0] << "\n"; 
    std::cout << "Tail result (C[last]): " << C[N-1] << "\n";

    return 0;
}

