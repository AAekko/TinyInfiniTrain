# TinyInfiniTrain 作业报告

## 一、test 通过截图
<img width="1053" height="664" alt="image" src="https://github.com/user-attachments/assets/2ac35c5b-4201-4438-92f3-c68c4563f3c3" />


## 二、作业步骤

> 将代码填入下面代码块中指定位置，并详细描述完成该作业的解决思路和遇到的问题。

### 作业一：autograd机制调用Neg kernel的实现

难度：⭐

对应测例：`TEST(ElementwiseTest, NegForward)`，`TEST(ElementwiseTest, NegBackward)`

需要实现的代码块位置：`infini_train/src/autograd/elementwise.cc`

```c++
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    // =================================== 作业 ===================================
    // TODO：通过Dispatcher获取设备专属kernel，对输入张量进行取反操作
    // NOTES: 依赖test_dispatcher，Neg kernel实现已给出
    // =================================== 作业 ===================================
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];
    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}


std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // =================================== 作业 ===================================
    // TODO：通过Dispatcher获取设备专属的反向传播kernel，计算梯度
    // NOTES: 依赖test_dispatcher，Neg的kernel实现已给出
    // =================================== 作业 ===================================
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];
    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output)};
}
```

#### 解决思路


#### 遇到问题



### 作业二：实现矩阵乘法

难度：⭐⭐

#### CPU实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiply)`，`TEST(MatmulTest, BatchedMatrixMultiply)`, `TEST(MatmulTest, BackwardPass)`

需要实现的代码块位置：`infini_train/src/kernels/cpu/linear.cc`

```c++
    std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
        // =================================== 作业 ===================================
        // TODO：实现CPU上的矩阵乘法前向计算
        // REF:
        // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_EQ(input_dims.size(), other_dims.size());
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.back(), other_dims[other_dims.size() - 2]);
    for (int64_t idx = 0; idx < input_dims.size() - 2; ++idx) { CHECK_EQ(input_dims[idx], other_dims[idx]); }

    auto output_dims = input_dims;
    output_dims.back() = other_dims.back();
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    const int64_t batch_size
        = std::accumulate(input_dims.begin(), input_dims.end() - 2, 1, std::multiplies<int64_t>());
    const int64_t rows = input_dims[input_dims.size() - 2];
    const int64_t inner = input_dims.back();
    const int64_t cols = other_dims.back();
    using MatrixMap = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    using MutableMatrixMap = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        MatrixMap input_matrix(static_cast<const float *>(input->DataPtr()) + batch * rows * inner, rows, inner);
        MatrixMap other_matrix(static_cast<const float *>(other->DataPtr()) + batch * inner * cols, inner, cols);
        MutableMatrixMap output_matrix(static_cast<float *>(output->DataPtr()) + batch * rows * cols, rows, cols);
        output_matrix = input_matrix * other_matrix;
    }
    return output;
    }

    std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
                    const std::shared_ptr<Tensor> &grad_output) {
        // =================================== 作业 ===================================
        // TODO：实现CPU上的矩阵乘法反向传播
        // REF:
        // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_EQ(input_dims.size(), other_dims.size());
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.back(), other_dims[other_dims.size() - 2]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32, other->GetDevice());
    const int64_t batch_size
        = std::accumulate(input_dims.begin(), input_dims.end() - 2, 1, std::multiplies<int64_t>());
    const int64_t rows = input_dims[input_dims.size() - 2];
    const int64_t inner = input_dims.back();
    const int64_t cols = other_dims.back();
    using MatrixMap = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    using MutableMatrixMap = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        MatrixMap input_matrix(static_cast<const float *>(input->DataPtr()) + batch * rows * inner, rows, inner);
        MatrixMap other_matrix(static_cast<const float *>(other->DataPtr()) + batch * inner * cols, inner, cols);
        MatrixMap grad_output_matrix(
            static_cast<const float *>(grad_output->DataPtr()) + batch * rows * cols, rows, cols);
        MutableMatrixMap grad_input_matrix(
            static_cast<float *>(grad_input->DataPtr()) + batch * rows * inner, rows, inner);
        MutableMatrixMap grad_other_matrix(
            static_cast<float *>(grad_other->DataPtr()) + batch * inner * cols, inner, cols);
        grad_input_matrix = grad_output_matrix * other_matrix.transpose();
        grad_other_matrix = input_matrix.transpose() * grad_output_matrix;
    }
    return {grad_input, grad_other};
    }
```

#### CUDA实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiplyCuda)`,`TEST(MatmulTest, BatchedMatrixMultiplyCuda)`,`TEST(MatmulTest, BackwardPassCuda)`

需要实现的代码块位置：`infini_train/src/kernels/cuda/linear.cu`

```c++
    std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
        // =================================== 作业 ===================================
        // TODO：实现CUDA上的矩阵乘法前向计算
        // REF:
        // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_EQ(input_dims.size(), other_dims.size());
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.back(), other_dims[other_dims.size() - 2]);
    for (int64_t idx = 0; idx < input_dims.size() - 2; ++idx) { CHECK_EQ(input_dims[idx], other_dims[idx]); }

    auto output_dims = input_dims;
    output_dims.back() = other_dims.back();
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());
    const int64_t batch_size
        = std::accumulate(input_dims.begin(), input_dims.end() - 2, 1, std::multiplies<int64_t>());
    const int64_t rows = input_dims[input_dims.size() - 2];
    const int64_t inner = input_dims.back();
    const int64_t cols = other_dims.back();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, rows, inner, &alpha,
        static_cast<const float *>(other->DataPtr()), cols, inner * cols,
        static_cast<const float *>(input->DataPtr()), inner, rows * inner, &beta,
        static_cast<float *>(output->DataPtr()), cols, rows * cols, batch_size));
    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
    }

    std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
                    const std::shared_ptr<Tensor> &grad_output) {
        // =================================== 作业 ===================================
        // TODO：实现CUDA上的矩阵乘法反向传播
        // REF:
        // =================================== 作业 ===================================
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    CHECK_EQ(input_dims.size(), other_dims.size());
    CHECK_GE(input_dims.size(), 2);
    CHECK_EQ(input_dims.back(), other_dims[other_dims.size() - 2]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32, other->GetDevice());
    const int64_t batch_size
        = std::accumulate(input_dims.begin(), input_dims.end() - 2, 1, std::multiplies<int64_t>());
    const int64_t rows = input_dims[input_dims.size() - 2];
    const int64_t inner = input_dims.back();
    const int64_t cols = other_dims.back();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, inner, rows, cols, &alpha,
        static_cast<const float *>(other->DataPtr()), cols, inner * cols,
        static_cast<const float *>(grad_output->DataPtr()), cols, rows * cols, &beta,
        static_cast<float *>(grad_input->DataPtr()), inner, rows * inner, batch_size));
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, cols, inner, rows, &alpha,
        static_cast<const float *>(grad_output->DataPtr()), cols, rows * cols,
        static_cast<const float *>(input->DataPtr()), inner, rows * inner, &beta,
        static_cast<float *>(grad_other->DataPtr()), cols, inner * cols, batch_size));
    CUBLAS_CHECK(cublasDestroy(handle));
    return {grad_input, grad_other};
    }
```

#### 解决思路



#### 遇到问题



### 作业三：实现Adam优化器

难度：⭐

#### CPU实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdate)`,`TEST(AdamOptimizerTest, MomentumAccumulation)`

代码位置：infini_train/src/kernels/cpu/accumulate_grad.cc

```c++
void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF: 
    // =================================== 作业 ===================================
    const float bias_correction1 = 1.0f - std::pow(beta1, t);
    const float bias_correction2 = 1.0f - std::pow(beta2, t);
    const auto *grad_ptr = static_cast<const float *>(grad->DataPtr());
    auto *param_ptr = static_cast<float *>(param->DataPtr());
    auto *m_ptr = static_cast<float *>(m->DataPtr());
    auto *v_ptr = static_cast<float *>(v->DataPtr());
    for (int64_t idx = 0; idx < grad->NumElements(); ++idx) {
        m_ptr[idx] = beta1 * m_ptr[idx] + (1.0f - beta1) * grad_ptr[idx];
        v_ptr[idx] = beta2 * v_ptr[idx] + (1.0f - beta2) * grad_ptr[idx] * grad_ptr[idx];
        const float m_hat = m_ptr[idx] / bias_correction1;
        const float v_hat = v_ptr[idx] / bias_correction2;
        param_ptr[idx] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}
```

#### CUDA实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdateCuda)`,`TEST(AdamOptimizerTest, MomentumAccumulationCuda)`

代码位置：infini_train/src/kernels/cuda/accumulate_grad.cu

```c++
void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF: 
    // =================================== 作业 ===================================
    size_t num_elements = grad->NumElements();
    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(
        static_cast<const float *>(grad->DataPtr()), static_cast<float *>(param->DataPtr()),
        static_cast<float *>(m->DataPtr()), static_cast<float *>(v->DataPtr()), learning_rate, beta1, beta2, eps, t,
        num_elements);
}
```

#### 解决思路



#### 遇到问题



### 作业四：实现Tensor基础操作

#### 实现Tensor的Flatten操作

难度：⭐

对应测例：`TEST(TensorTransformTest, Flatten2DTo1D)`,`TEST(TensorTransformTest, FlattenWithRange) `,`TEST(TensorTransformTest, FlattenNonContiguous)`

代码位置：infini_train/src/tensor.cc

```c++
std::shared_ptr<Tensor> Tensor::Flatten(int64_t start, int64_t end) {
    // =================================== 作业 ===================================
    // TODO：实现张量扁平化操作，将指定维度范围[start, end]内的所有维度合并为一个维度
    // HINT: 
    // =================================== 作业 ===================================
    const int64_t num_dims = dims_.size();
    if (start < 0) { start += num_dims; }
    if (end < 0) { end += num_dims; }
    CHECK_GE(start, 0);
    CHECK_LE(start, end);
    CHECK_LT(end, num_dims);

    std::vector<int64_t> new_shape(dims_.begin(), dims_.begin() + start);
    new_shape.push_back(std::accumulate(dims_.begin() + start, dims_.begin() + end + 1, 1,
                                        std::multiplies<int64_t>()));
    new_shape.insert(new_shape.end(), dims_.begin() + end + 1, dims_.end());
    return Contiguous()->View(new_shape);
}
```

#### 实现Tensor的反向传播机制

难度：⭐

对应测例：`TEST(TensorAutogradTest, BackwardComputesGradient)`,`TEST(TensorAutogradTest, BackwardWithMultipleOutputs)`

代码位置：infini_train/src/tensor.cc

```c++
void Tensor::Backward(std::shared_ptr<Tensor> gradient, bool retain_graph, bool create_graph) const {
    // =================================== 作业 ===================================
    // TODO：实现自动微分反向传播
    // 功能描述：1. 计算当前张量对叶子节点的梯度    2. 支持多输出场景的梯度累加
    // HINT: 
    // =================================== 作业 ===================================
    CHECK(grad_fn_) << "Cannot call Backward on a tensor that has no grad function";
    if (!gradient) {
        CHECK_EQ(NumElements(), 1) << "Gradient can be implicitly created only for scalar outputs";
        gradient = std::make_shared<Tensor>(dims_, dtype_, GetDevice());
        gradient->Fill<float>(1.0f);
    }
    CHECK(gradient->Dims() == dims_);
    grad_fn_->BackwardPartial(gradient, output_idx_);
}
```

#### 解决思路



#### 遇到问题



### 作业五 注册算子kernel的实现

难度：⭐⭐⭐

对应测例：`TEST(DispatcherTest, RegisterAndGetKernel)`,`TEST(DispatcherTest, DuplicateRegistration)`,`TEST(DispatcherTest, GetNonexistentKernel)`

代码位置：infini_train/include/dispatcher.h

```c++
template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
    // =================================== 作业 ===================================
    // TODO：实现通用kernel调用接口
    // 功能描述：将存储的函数指针转换为指定类型并调用
    // HINT: 
    // =================================== 作业 ===================================
    using FuncT = RetT (*)(ArgsT...);
    // TODO: 实现函数调用逻辑
    return reinterpret_cast<FuncT>(func_ptr_)(args...);
}

template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
    // =================================== 作业 ===================================
    // TODO：实现kernel注册机制
    // 功能描述：将kernel函数与设备类型、名称绑定
    // =================================== 作业 ===================================
    CHECK(!key_to_kernel_map_.contains(key))
        << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
    key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
}

#define REGISTER_KERNEL(device, kernel_name, kernel_func) 
    // =================================== 作业 ===================================
    // TODO：实现自动注册宏
    // 功能描述：在全局静态区注册kernel，避免显式初始化代码
    // =================================== 作业 ===================================
    static const bool REGISTER_KERNEL_CONCAT(kernel_registered_, __COUNTER__) = []() {                                \
        infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func);                            \
        return true;                                                                                                   \
    }();
```

#### 解决思路



#### 遇到问题



### 作业六：实现GPT-2整体训练

难度：⭐⭐⭐⭐

对应测例：`TEST_F(GPT2TrainingTest, LogitsConsistency)`

#### 训练过程logits对比

完成以上所有作业，补齐训练框架的所有实现，理论上`TEST_F(GPT2TrainingTest, LogitsConsistency)`可以通过，在用例中判断比较预置的值和单步正向传播计算结果是否在误差允许范围内相等。

#### 数据读取实现

代码位置：example/common/tiny_shakespeare_dataset.cc

```c++
TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    CHECK_GT(sequence_length, 0);
    if (!std::filesystem::exists(path)) {
        LOG(FATAL) << "File not found: " << path;
    }

    std::ifstream ifs(path, std::ios::binary);
    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    const auto magic = BytesToType<int32_t>(header, 0);
    CHECK(kTypeMap.contains(magic));
    const auto type = kTypeMap.at(magic);
    const auto num_tokens = BytesToType<uint32_t>(header, 8);

    auto token_data = ReadSeveralBytesFromIfstream(num_tokens * kTypeToSize.at(type), &ifs);
    auto tensor = infini_train::Tensor({static_cast<int64_t>(num_tokens)}, DataType::kINT64);
    auto *tensor_data = static_cast<int64_t *>(tensor.DataPtr());
    if (type == TinyShakespeareType::kUINT16) {
        for (size_t idx = 0; idx < num_tokens; ++idx) {
            tensor_data[idx] = BytesToType<uint16_t>(token_data, idx * sizeof(uint16_t));
        }
    } else {
        for (size_t idx = 0; idx < num_tokens; ++idx) {
            tensor_data[idx] = BytesToType<uint32_t>(token_data, idx * sizeof(uint32_t));
        }
    }

    return {type,
            {static_cast<int64_t>((num_tokens - 1) / sequence_length + 1), static_cast<int64_t>(sequence_length)},
            std::move(tensor)};
}

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      num_samples_(text_file_.dims[0] - 1) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
}
```

#### Tokenizer功能实现

代码位置：example/common/tokenizer.cc

```c++
Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File not found: " << filepath;
    }

    std::ifstream ifs(filepath, std::ios::binary);
    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    magic_number_ = BytesToType<uint32_t>(header, 0);
    CHECK(kEotMap.contains(magic_number_));
    const auto version = BytesToType<Version>(header, 4);
    CHECK(version == Version::kV1 || version == Version::kV2);
    vocab_size_ = BytesToType<uint32_t>(header, 8);
    token_table_.reserve(vocab_size_);
    for (uint32_t idx = 0; idx < vocab_size_; ++idx) {
        const auto token_length = BytesToType<uint8_t>(ReadSeveralBytesFromIfstream(1, &ifs), 0);
        const auto token_bytes = ReadSeveralBytesFromIfstream(token_length, &ifs);
        token_table_.emplace_back(reinterpret_cast<const char *>(token_bytes.data()), token_bytes.size());
    }
    eot_token_ = kEotMap.at(magic_number_);
}
```

```c++
std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    CHECK_LT(token_id, token_table_.size());
    return token_table_[token_id];
}
```

```c++
void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t kRngState = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
    if (t == prompt_len) { kRngState = infini_train::kRngState; }
        auto logits = model.Forward({x})[0];
        auto probabilities = nn::function::Softmax(logits, -1)->To(Device(DeviceType::kCPU, 0));
        const int64_t vocab_size = probabilities.Dims().back();
        const int64_t time_step = std::min<int64_t>(t - 1, sequence_length - 1);
        float *probabilities_data = static_cast<float *>(probabilities.DataPtr()) + time_step * vocab_size;
        const int next_token = SampleMult(probabilities_data, vocab_size, RandomF32(kRngState));
        std::cout << Decode(next_token);

        auto x_cpu = x->To(Device(DeviceType::kCPU, 0));
        auto *x_data = static_cast<int64_t *>(x_cpu.DataPtr());
        for (uint32_t batch = 0; batch < batch_size; ++batch) {
            if (t < sequence_length) {
                x_data[batch * sequence_length + t] = next_token;
            } else {
                std::memmove(x_data + batch * sequence_length, x_data + batch * sequence_length + 1,
                             (sequence_length - 1) * sizeof(int64_t));
                x_data[(batch + 1) * sequence_length - 1] = next_token;
            }
        }
        x = std::make_shared<infini_train::Tensor>(x_cpu.To(device));
    }
    std::cout << std::endl;
}
```

#### 解决思路



#### 遇到问题
多次测试总是在模型输出的第 385973 个数据点上出现误差在0.002左右，达不到0.001的精度
