# Ubuntu 编译环境配置指南

## 安装必要的编译工具

```bash
# 更新包管理器
sudo apt-get update

# 安装基础编译工具
sudo apt-get install -y build-essential

# 安装 CMake (可选)
sudo apt-get install -y cmake

# 安装 Boost 库
sudo apt-get install -y libboost-regex-dev
```

## 验证安装

```bash
# 检查 g++ 版本
g++ --version

# 检查 make 版本
make --version

# 检查 Boost 库
dpkg -l | grep libboost-regex
```

## 编译项目

### 方法 1: 使用 Makefile

```bash
cd /home/qww/work/qwen3.cpp
make
```

### 方法 2: 使用 CMake

```bash
cd /home/qww/work/qwen3.cpp
mkdir build
cd build
cmake ..
make
```

### 方法 3: 直接使用 g++

```bash
cd /home/qww/work/qwen3.cpp
g++ -std=c++17 -Wall -Wextra -O2 -I. -c tokenizer.cpp -o tokenizer.o
g++ -std=c++17 -Wall -Wextra -O2 -I. -c main.cpp -o main.o
g++ -std=c++17 -o qwen3 main.o tokenizer.o -lboost_regex
```

## 运行程序

```bash
./qwen3 [tokenizer.json路径] [model.safetensors路径]
```

## 常见问题

### 1. 找不到 Boost 库

如果编译时提示找不到 Boost 库，请确保已安装：
```bash
sudo apt-get install libboost-all-dev
```

### 2. C++17 支持

确保 g++ 版本 >= 7.0，支持 C++17 标准。

### 3. 链接错误

如果出现链接错误，尝试添加 `-lboost_regex` 链接选项。
