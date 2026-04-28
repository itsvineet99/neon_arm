#include <iostream>
#include <arm_neon.h>
#include <vector>
#include <chrono>

float vector_dot(const float* __restrict A, const float* __restrict B, size_t n) {
    size_t i = 0;

    // Accumulate into 4 parallel lanes
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4) {
        // loading A and B in neon registers
        float32x4_t va = vld1q_f32(&A[i]);
        float32x4_t vb = vld1q_f32(&B[i]);

        // neon instuction to get dot product
        sum_vec = vmlaq_f32(sum_vec, va, vb);
    }

    // Horizontal Add: Sum the 4 lanes of sum_vec into a single scalar float
    float total_sum = vaddvq_f32(sum_vec);

    // when n isn't perfictly divided by 4
    for (; i < n; i++) {
        total_sum += A[i] * B[i];
    }

    return total_sum;
}

float naive_dot(const float* A, const float* B, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i ++) {
        sum += A[i] * B[i];
    }
    return sum;
}

int main() {
    size_t N; 
    std::cout << "Enter vector size: ";
    
    if (std::cin >> N) {
        std::cout << "You entered: " << N << "\n";
    } else {
        std::cout << "Invalid input. Please enter a positive number.\n";
    }

    std::vector<float> A(N, 3.0f);
    std::vector<float> B(N, 2.5f);

    float result_naive = 0.0f;
    float result_vec = 0.0f;

    // --- NAIVE RUN ---
    // Warmup
    for (int i = 0; i < 10; i++) {
        // Volatile cast prevents the compiler from optimizing the loop away entirely
        volatile float temp = naive_dot(A.data(), B.data(), N);
        (void)temp;
    }

    auto n_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; i++) {
        result_naive = naive_dot(A.data(), B.data(), N);
    }
    auto n_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> n_time = (n_end - n_start) / 20.0;

    // --- VECTOR RUN ---
    // Warmup
    for (int i = 0; i < 10; i++) {
        volatile float temp = vector_dot(A.data(), B.data(), N);
        (void)temp;
    }

    auto v_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; i++) {
        result_vec = vector_dot(A.data(), B.data(), N);
    }
    auto v_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> v_time = (v_end - v_start) / 20.0;

    // --- RESULTS ---
    std::cout << "\nResults averaged over 20 runs:\n";
    std::cout << "Vector implementation computed " << N << " element dot product in " << v_time.count() << " ms\n";
    std::cout << "Naive implementation computed  " << N << " element dot product in " << n_time.count() << " ms\n";
    
    std::cout << "\nValidation:\n";
    std::cout << "Vector Result: " << result_vec << "\n";
    std::cout << "Naive Result:  " << result_naive << "\n";
    std::cout << "Math match? " << (result_vec == result_naive ? "YES" : "NO") << "\n";

    return 0;
}
