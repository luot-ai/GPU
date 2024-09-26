#ifndef BMM_HPP
#define BMM_HPP

void Bmm_cpu(
    std::vector<float> input_A,
    std::vector<float> input_B,
    std::vector<float> &output,
    int M_A,
    int K_A,
    int K_B,
    int N_B,
    int BatchSize = 1)
{
    // 输入规模：BatchSize * M_A * K_A, BatchSize * K_B * N_B
    // 输出规模：BatchSize * M_A * N_B
    if (input_A.size() != BatchSize * M_A * K_A)
    {
        throw "Bmm_cpu input_A size error";
    }

    if (input_B.size() != BatchSize * K_B * N_B)
    {
        throw "Bmm_cpu input_B size error";
    }

    if (output.size() != BatchSize * M_A * N_B)
    {
        throw "Bmm_cpu output size error";
    }

    for (int iter_batch = 0; iter_batch < BatchSize; iter_batch++)
    {
        for (int iter_M_A = 0; iter_M_A < M_A; iter_M_A++)
        {
            for (int iter_N_B = 0; iter_N_B < N_B; iter_N_B++)
            {
                output[iter_batch * M_A * N_B + iter_M_A * N_B + iter_N_B] = 0;
                for (int iter_K_A = 0; iter_K_A < K_A; iter_K_A++)
                {
                    output[iter_batch * M_A * N_B + iter_M_A * N_B + iter_N_B] +=
                        input_A[iter_batch * M_A * K_A + iter_M_A * K_A + iter_K_A] *
                        input_B[iter_batch * K_B * N_B + iter_K_A * N_B + iter_N_B];
                }
            }
        }
    }
}

#endif 