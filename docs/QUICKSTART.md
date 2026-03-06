# 快速开始指南

## 1. 安装依赖（Ubuntu/Debian）

```bash
sudo apt-get update
sudo apt-get install -y build-essential libboost-regex-dev
```

## 2. 编译

```bash
cd /home/qww/work/qwen3.cpp
make
```

或者使用 CMake：

```bash
mkdir build && cd build
cmake ..
make
```

## 3. 准备模型文件

确保你有以下文件：
- `tokenizer.json` - Qwen3 的 tokenizer 配置
- `model.safetensors` - Qwen3 的模型权重

## 4. 运行

```bash
./qwen3 /path/to/tokenizer.json /path/to/model.safetensors
```

## 项目结构

```
qwen3.cpp/
├── main.cpp           - 主程序（60 行）
├── tokenizer.cpp      - Tokenizer 实现（200 行）
├── tokenizer.h        - Tokenizer 头文件（30 行）
├── type.h             - 数据结构定义（60 行）
├── nlohmann/          - JSON 库
├── Makefile           - Make 构建
├── CMakeLists.txt     - CMake 构建
├── README.md          - 项目说明
├── INSTALL.md         - 详细安装指南
├── REFACTOR.md        - 重构总结
└── .gitignore         - Git 配置
```

## 代码统计

- 总代码行数: ~395 行（不含 nlohmann JSON 库）
- 核心功能: Tokenizer (BPE 编码)
- 待实现: Transformer 层、推理引擎

## 主要改进

✅ 移除 Windows 特定代码
✅ 跨平台支持（Linux/macOS/Windows）
✅ 代码模块化
✅ 清理重复定义
✅ 标准 C++17
✅ 完整的构建系统

## 下一步

1. 实现 Transformer 层
2. 实现 RoPE 位置编码
3. 实现 Attention 机制
4. 实现 FFN 层
5. 实现推理引擎
