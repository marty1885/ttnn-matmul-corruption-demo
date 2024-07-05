#  ttnn-matmul-corruption-demo

This repository contains code that reflects [#9849](https://github.com/tenstorrent/tt-metal/issues/9849) in the tt-metal repository.

## How to compile

```bash
# The regular env variables needed in TTNN
export ARCH_NAME=grayskull
export TT_METAL_HOME=/path/to/your/tt-metal

# Build
mkdir build
cd build
cmake .. -DGGML_METALIUM=ON -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
make -j
```

## How to run

```bash
./ttnn-matmul-corruption-demo
```

## Output

```
                 Device | INFO     | Opening user mode device driver
2024-07-05 14:10:10.282 | INFO     | SiliconDriver   - Detected 1 PCI device : {0}
                  Metal | INFO     | Initializing device 0. Program cache is NOT enabled
                  Metal | INFO     | AI CLK for device 0 is:   1000 MHz
                  Metal | INFO     | Enabling program cache on device 0
                  Verif | INFO     | Created a random vector of size 81920
                  Verif | INFO     | Created a random vector of size 163840
                  Verif | INFO     | Created a random vector of size 1024
.... (printing numbers)
....
....


1.21937e+29 NaN or corrupted value detected at index 10240
[1]    3972 IOT instruction (core dumped)  ./ttnn-matmul-corruption-demo
```