# ONNX to Torch lowering

ONNX operators are defined [here](https://github.com/onnx/onnx/blob/main/docs/Operators.md).
Here torch means ATen operators. ATen is fundamentally a tensor library, on top of which almost all other Python and C++ interfaces in PyTorch are built. The list of ATen operators can be found [here](https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir/dialects/torch/importer/jit_ir/build_tools/torch_ods_gen.py).

We lower from certain version of ONNX operator to unfixed version of ATen operators. ONNX has a spec of the operator for each revision etc. We can see various revisions for ONNX [here](https://github.com/onnx/onnx/blob/main/docs/Operators.md#aionnx-default). Torch does not have a fixed version of ATen operators to which we convert. The list of operators within ATen do not seem complete either. We have lots of instances in which there is a PyTorch api but does not have a corresponding ATen operator (EluOp, pixel_shuffle, pixel_unshuffle etc.).

## Direct conversion

There are cases where there is a direct conversion from ONNX operator to Torch, they accept the same (in dimensions and types) operands and return the same output. In subset of these cases, we are able to write generic C++ templates that generate code for lowering from ONNX to Torch. Examples where there is a direct mapping are:

### Unary operator
- `ONNXSinOp` -> `AtenSinOp`
- `ONNXExpOp` -> `AtenExpOp`
- ...
- Implementation of unary template can be found in `onnx-mlir/src/Conversion/ONNXToTorch/NN/ElemenwiseOp.cpp`

### Binary operator
- `ONNXMatMulOp` ->  `AtenMatmulOp`
- Implementation of binary template can be found in `onnx-mlir/src/Conversion/ONNXToTorch/NN/BinaryOps.cpp`

### Varidic operator
- This has not been implemented yet, but will be implemented in the near future.

## Indirect conversion
In cases where direct conversion does not exist, we try to perform the conversion with elementary operators such as `Add`, `Multiply` etc. Examples are as follows:

- GemmOp
- SoftMax
- etc.
