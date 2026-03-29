# 第 2 章：环境集成 —— 搭建你的 C++ 开发工具链

在正式开始编码之前，我们需要先搭建一个干净、高效的开发环境。这一章会带你完成从零到一的环境配置，确保后续的开发过程顺畅无阻。

### 2.1 系统要求

我们的项目对硬件和系统的要求非常友好，一台普通的笔记本电脑就能胜任：

**硬件要求**：
- **CPU**：任何支持 x86_64 的现代处理器（Intel/AMD 均可）
- **内存**：至少 4GB RAM（推荐 8GB 以上）
- **硬盘**：约 2GB 空闲空间（用于存放模型权重和编译产物）

**操作系统**：
- Linux（推荐 Ubuntu 20.04 或更高版本）
- macOS（10.15 或更高版本）
- Windows（通过 WSL2 或 MinGW）

**编译器要求**：
- GCC 7.0+ 或 Clang 5.0+（需要支持 C++17 标准）
- CMake 3.10+

### 2.2 安装必要的工具

#### 2.2.1 安装编译器和构建工具

**Ubuntu/Debian 系统**：
```bash
sudo apt update
sudo apt install -y build-essential cmake git
```

**macOS 系统**：
```bash
# 安装 Xcode Command Line Tools
xcode-select --install

# 安装 CMake（通过 Homebrew）
brew install cmake
```

**Windows 系统**：
推荐使用 WSL2（Windows Subsystem for Linux），然后按照 Ubuntu 的步骤安装。

#### 2.2.2 安装 Boost 库

我们的项目依赖 Boost 库中的正则表达式模块（用于分词器的实现）。

**Ubuntu/Debian 系统**：
```bash
sudo apt install -y libboost-regex-dev
```

**macOS 系统**：
```bash
brew install boost
```

**验证安装**：
```bash
# 检查 Boost 版本
dpkg -s libboost-regex-dev | grep Version  # Ubuntu
brew info boost                       # macOS
```

### 2.3 创建项目目录结构

现在让我们创建一个清晰的项目目录结构。打开终端，执行以下命令：

```bash
# 创建项目根目录
mkdir -p qwen3.cpp
cd qwen3.cpp

# 创建源代码目录
mkdir -p src

# 创建第三方库目录
mkdir -p thirdparty/nlohmann
```

完成后，你的项目结构应该是这样的：

```
qwen3.cpp/
├── src/                    # 源代码目录
└── thirdparty/             # 第三方库目录
    └── nlohmann/           # JSON 解析库
```

### 2.4 配置第三方依赖

我们的项目只依赖一个轻量级的第三方库：**nlohmann/json**，用于解析模型配置文件和分词器配置。

#### 2.4.1 下载 nlohmann/json

这是一个 header-only 的 JSON 库，只需要下载两个头文件即可：

```bash
cd thirdparty/nlohmann

# 下载 json.hpp（主文件）
wget https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp

# 下载 json_fwd.hpp（前向声明文件）
wget https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json_fwd.hpp

cd ../..
```

如果你的系统没有 `wget`，可以使用 `curl`：

```bash
curl -O https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp
curl -O https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json_fwd.hpp
```

**为什么选择 nlohmann/json？**
- **Header-only**：无需编译，直接包含即可使用
- **现代 C++ 风格**：API 简洁优雅，符合 C++11/14/17 标准
- **轻量级**：单文件实现，不会给项目增加额外负担

### 2.5 编写 CMakeLists.txt

CMake 是我们的构建系统，它能够跨平台地管理编译过程。在项目根目录创建 `CMakeLists.txt` 文件：

```bash
cd qwen3.cpp
touch CMakeLists.txt
```

在 `CMakeLists.txt` 中写入以下内容：

```cmake
cmake_minimum_required(VERSION 3.10)
project(qwen3)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译模式为 Release（优化性能）
set(CMAKE_BUILD_TYPE Release)

# 查找 Boost 库（需要 regex 组件）
find_package(Boost REQUIRED COMPONENTS regex)

# 创建可执行文件：qwen3（测试程序）
add_executable(qwen3
    src/main.cpp
)

# 设置头文件搜索路径
target_include_directories(qwen3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 链接 Boost 库
target_link_libraries(qwen3 PRIVATE Boost::regex)
```

**CMakeLists.txt 关键点解读**：

1. **C++17 标准**：我们使用 C++17 的一些现代特性（如结构化绑定、`std::optional` 等）。
2. **Release 模式**：默认使用 Release 模式编译，启用编译器优化，提升运行性能。
3. **Boost::regex**：用于分词器中的正则表达式匹配（BPE 算法需要）。
4. **可执行文件**：
   - `qwen3`：用于测试和验证各个模块

### 2.6 验证环境配置

现在让我们验证一下环境是否配置正确。创建一个简单的测试程序：

```bash
cd src
touch main.cpp
```

在 `main.cpp` 中写入以下测试代码：

```cpp
#include <iostream>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

int main() {
    // 测试 JSON 库
    json config = {
        {"model", "Qwen3-0.6B"},
        {"vocab_size", 151936},
        {"hidden_size", 896}
    };

    std::cout << "Environment Test Passed!" << std::endl;
    std::cout << "Model: " << config["model"] << std::endl;
    std::cout << "Vocab Size: " << config["vocab_size"] << std::endl;

    return 0;
}
```

现在回到项目根目录，创建构建目录并编译：

```bash
cd ..
mkdir build
cd build

# 配置项目
cmake ..

# 编译
make
```

如果一切顺利，你应该看到类似这样的输出：

```
-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
...
-- Found Boost: /usr/lib/x86_64-linux-gnu/cmake/Boost-1.74.0/BoostConfig.cmake (found version "1.74.0")
-- Configuring done
-- Generating done
-- Build files have been written to: /path/to/qwen3.cpp/build
[ 33%] Building CXX object CMakeFiles/qwen3.dir/src/main.cpp.o
[ 66%] Linking CXX executable qwen3
[100%] Built target qwen3
```

运行测试程序：

```bash
./qwen3
```

预期输出：

```
Environment Test Passed!
Model: "Qwen3-0.6B"
Vocab Size: 151936
```

### 2.7 项目最终目录结构

完成环境配置后，你的项目目录应该是这样的：

```
qwen3.cpp/
├── CMakeLists.txt          # CMake 构建配置文件
├── build/                  # 构建目录（编译产物）
│   ├── qwen3               # 测试可执行文件
├── src/                    # 源代码目录
│   ├── main.cpp            # 测试程序入口
│   ├── tensor.h            # 张量类（第 3 章）
│   ├── tokenizer.h         # 分词器头文件（第 4 章）
│   ├── tokenizer.cpp       # 分词器实现（第 4 章）
│   ├── operator.hpp        # 核心算子实现（第 5 ~ 7 章）
│   ├── qwen3_model.h       # 模型头文件（第 8 章）
│   ├── qwen3_model.cpp     # 模型实现（第 8 章）
│   └── logger.h            # 日志工具（辅助）
└── thirdparty/             # 第三方库目录
    └── nlohmann/           # JSON 解析库
        ├── json.hpp
        └── json_fwd.hpp
```

恭喜！你已经成功搭建了开发环境。下一章，我们将实现第一个核心组件：**张量（Tensor）**，它是整个推理引擎的数据基础。

