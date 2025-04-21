# 调试测试提示

## 如何运行、执行或调试一个特定的测试，而不运行其他任何测试，以缩短反馈循环？

在 `scripts` 文件夹中有一个名为 `debug-test.sh` 的脚本，它的参数可以接受一个正则表达式（REGEX）和一个可选的测试编号。

例如，运行以下命令将输出一个交互式列表，您可以从中选择一个测试。其形式如下：

```bash
debug-test.sh [OPTION]... <test_regex> <test_number>
```

它将为您构建并在调试器中运行测试。

要仅执行测试并返回 PASS 或 FAIL 消息，请运行：

```bash
./scripts/debug-test.sh test-tokenizer
```

要在 GDB 中测试，请使用 `-g` 标志以启用 GDB 测试模式。

```bash
./scripts/debug-test.sh -g test-tokenizer

# 进入调试器后，即提示符处，设置断点可以如下进行：
>>> b main
```

为了加快测试循环，如果您知道测试编号，可以直接运行它，例如：

```bash
./scripts/debug-test.sh test 23
```

如需进一步参考，请使用 `debug-test.sh -h` 打印帮助信息。

&nbsp;

### 脚本如何工作？
如果您想单独使用脚本中的概念，以下是一些重要步骤的简要说明。

#### 第一步：重置并设置文件夹上下文

从此存储库的根目录开始，创建 `build-ci-debug` 作为我们的构建上下文。

```bash
rm -rf build-ci-debug && mkdir build-ci-debug && cd build-ci-debug
```

#### 第二步：设置构建环境并编译测试二进制文件

在调试模式下设置并触发构建。您可以根据需要调整参数，但这里使用的是合理的默认值。

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DLLAMA_CUDA=1 -DLLAMA_FATAL_WARNINGS=ON ..
make -j
```

#### 第三步：查找符合正则表达式的所有测试

此命令的输出将为您提供运行 GDB 所需的命令和参数。

* `-R test-tokenizer`：查找所有名为 `test-tokenizer*` 的测试文件（R=正则表达式）。
* `-N`："仅显示" 禁用测试执行，只显示可以传递给 GDB 的测试命令。
* `-V`：详细模式。

```bash
ctest -R "test-tokenizer" -V -N
```

这可能会返回类似以下的输出（重点关注关键行）：

```bash
...
1: Test command: ~/llama.cpp/build-ci-debug/bin/test-tokenizer-0 "~/llama.cpp/tests/../models/ggml-vocab-llama-spm.gguf"
1: Working Directory: .
Labels: main
  Test  #1: test-tokenizer-0-llama-spm
...
4: Test command: ~/llama.cpp/build-ci-debug/bin/test-tokenizer-0 "~/llama.cpp/tests/../models/ggml-vocab-falcon.gguf"
4: Working Directory: .
Labels: main
  Test  #4: test-tokenizer-0-falcon
...
```

#### 第四步：识别用于调试的测试命令

根据上面测试 #1 的信息，我们可以确定以下两个相关信息：
* 测试二进制文件：`~/llama.cpp/build-ci-debug/bin/test-tokenizer-0`
* 测试 GGUF 模型：`~/llama.cpp/tests/../models/ggml-vocab-llama-spm.gguf`

#### 第五步：在测试命令上运行 GDB

根据上面的 ctest 'test command' 报告，我们可以通过以下命令启动 GDB 会话：

```bash
gdb --args ${Test Binary} ${Test GGUF Model}
```

示例：

```bash
gdb --args ~/llama.cpp/build-ci-debug/bin/test-tokenizer-0 "~/llama.cpp/tests/../models/ggml-vocab-llama-spm.gguf"
```