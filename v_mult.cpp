#include <iostream>
#include <arm_neon.h>
#include <vector>
#include <chrono>

void vector_multiplication(const float* __restrict A, const float* __restrict B,
                            float* __restrict C, size_t n) {
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        // loading A and B in neon registers
        float32x4_t va = vld1q_f32(&A[i]);
        float32x4_t vb = vld1q_f32(&B[i]);

        // neon instuction to multiply them
        float32x4_t vc = vmulq_f32(va, vb);

        // neon instruction to save it to C
        vst1q_f32(&C[i], vc);
    }

    // when n isn't perfictly divided by 4
    for (; i < n; i++) {
        C[i] = A[i] * B[i];
    }
}

void naive_mult (const float* A, const float* B, float* C, size_t n) {
    for (size_t i = 0; i < n; i ++) {
        C[i] = A[i] * B[i];
    }
}

int main() {
    size_t N; // vector implementation works best for size 10,000 
    std::cout << "Enter vector size: ";
    
    if (std::cin >> N) {
        std::cout << "You entered: " << N << "\n";
    } else {
        std::cout << "Invalid input. Please enter a positive number.\n";
    }

    std::vector<float> A(N, 3.0f);
    std::vector<float> B(N, 2.5f);
    std::vector<float> C_vec(N, 0.0f);

    // different variables for both cause cache helps the implementations 
    // thats sencond in order in code and gives performance boost
    std::vector<float> P(N, 3.0f);
    std::vector<float> Q(N, 2.5f);
    std::vector<float> C_naive(N, 0.0f);

     // --- WARMUP RUN ---
    for (int i = 0; i < 10; i++) {
        naive_mult(P.data(), Q.data(), C_naive.data(), N);
    }

    auto n_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; i++) {
        naive_mult(P.data(), Q.data(), C_naive.data(), N);;
    }
    auto n_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> n_time = (n_end - n_start) / 20;

    // --- WARMUP RUN ---
    for (int i = 0; i < 10; i++) {
        vector_multiplication(A.data(), B.data(), C_vec.data(), N);
    }

    auto v_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; i++) {
        vector_multiplication(A.data(), B.data(), C_vec.data(), N);
    }
    auto v_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> v_time = (v_end - v_start) / 20;

    std::cout << "Vector implementation computed " << N << " multiplications in " << v_time.count() << " ms\n";
    std::cout << "Naive implementation computed " << N << " multiplications in " << n_time.count() << " ms\n";
    std::cout << "is result correct: " << (C_vec[0] == C_naive[0]) << "\n"; // Should be true

    return 0;
}
