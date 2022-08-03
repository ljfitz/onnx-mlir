/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===- Conv2D.cpp - Lowering Convolution Op -===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// ========================================================================
//
// This file lowers the ONNX Convolution Operators to Torch dialect.
//
//===-----------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

#ifdef _WIN32
#include <io.h>
#endif

/*
 * ONNX Conv operation
 * “The convolution operator consumes an input tensor and a filter,
 * and” “computes the output.
 *
 * Attributes:
   *	auto_pad	::mlir::StringAttr	string attribute
   * 	dilations	::mlir::ArrayAttr	64-bit integer array
   *	group		::mlir::IntegerAttr	64-bit signed integer
   *	kernel_shape	::mlir::ArrayAttr	64-bit integer array
   *	pads		::mlir::ArrayAttr	64-bit integer array
   *	strides		::mlir::ArrayAttr	64-bit integer array

 *Operands:
     *	X tensor of 16-bit/32-bit/64-bit float values or memref
            of any type values
     *	W tensor of 16-bit/32-bit/64-bit float values or memref
            of any type values
     *	B tensor of 16-bit/32-bit/64-bit float values or memref
            of any type values or none type
 *Results:
     *	Y  tensor of 16-bit/32-bit/64-bit float values or memref
     *	  of any type values or none type
 */

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

struct ONNXConvOpToTorchLowering : public ConversionPattern {
  ONNXConvOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXConvOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXConvOpAdaptor operandAdaptor(operands);
    ONNXConvOp op1 = llvm::dyn_cast<ONNXConvOp>(op);

    mlir::MLIRContext *context = op1.getContext();

    Value x = op1.X(); // ONNX operands
    Value w = op1.W(); // ONNX operands
    Value b = op1.B(); // ONNX operands
    bool biasIsNone = b.getType().isa<mlir::NoneType>();

    auto autopad = op1.auto_padAttr();          // ::mlir::StringAttr
    auto dilations = op1.dilationsAttr();       // ::mlir::ArrayAttr
    auto group = op1.groupAttr();               // ::mlir::IntegerAttr
    auto kernal_shape = op1.kernel_shapeAttr(); // ::mlir::ArrayAttr
    auto pads = op1.padsAttr();                 // ::mlir::ArrayAttr
    auto strides = op1.stridesAttr();           // ::mlir::ArrayAttr

    // NOTE: we would like if inferShapes() had filled in explicit padding
    // but currently inferShapes() does not do this for ConvOp (it does for
    // ConvTransposeOp). We have not implemented code for autopad so fail.
    if (autopad && autopad != "NOTSET")
      return rewriter.notifyMatchFailure(op, "padding must be explicit");

    // create vector of tensor list iterate through the ArrayAttribute
    // list.
    auto sintType = IntegerType::get(op1.getContext(), 64, IntegerType::SignednessSemantics::Signed);
    std::vector<Value> translatepadsList =
        createPadsArrayAttribute(pads, sintType, loc, rewriter);
    std::vector<Value> dilationonnxList =
        createArrayAttribute(dilations, sintType, loc, rewriter, 1);
    std::vector<Value> kernalshapeonnxList =
        createArrayAttribute(kernal_shape, sintType, loc, rewriter);
    std::vector<Value> stridesonnxList =
        createArrayAttribute(strides, sintType, loc, rewriter);

    // If group Value is null, assigning default value.
    Value groupTorchInt;
    if (group) {
      groupTorchInt = rewriter.create<ConstantIntOp>(loc, group);
    } else {
      // NOTE: we would like if inferShapes() had filled in default values
      // so we could assume `group` is always set, but currently inferShapes()
      // does not do this for ConvOp (it does for ConvTransposeOp).
      auto oneAttr = IntegerAttr::get(sintType, 1);
      groupTorchInt = rewriter.create<ConstantIntOp>(loc, oneAttr);
    }

    // create the Torch List type using above created vectors.
    Value stridesList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{stridesonnxList});

    Value dilationList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{dilationonnxList});

    Value padsList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{translatepadsList});

    // create a tensor types using onnx operands.
    TensorType xTensorType = x.getType().cast<TensorType>();
    TensorType wTensorType = w.getType().cast<TensorType>();
    TensorType opTensorType = op->getResult(0).getType().cast<TensorType>();

    auto xType = Torch::ValueTensorType::get(
        context, xTensorType.getShape(), xTensorType.getElementType());
    auto wType = Torch::ValueTensorType::get(
        context, wTensorType.getShape(), wTensorType.getElementType());
    auto resultType = Torch::ValueTensorType::get(op1.getContext(),
        opTensorType.getShape(), opTensorType.getElementType());

    auto xTorchTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, xType, x);
    auto wTorchTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, wType, w);
    Value bTorchTensor;
    if (biasIsNone) {
      bTorchTensor = rewriter.create<Torch::ConstantNoneOp>(loc);
    } else {
      TensorType bTensorType = b.getType().cast<TensorType>();
      auto bType = Torch::ValueTensorType::get(op1.getContext(),
          bTensorType.getShape(), bTensorType.getElementType());
      bTorchTensor =
          rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
              loc, bType, b);
    }

    // emit the Conv2d operation in Torch side using "AtenConv2dOp".
    Value result = rewriter.create<AtenConv2dOp>(loc, resultType, xTorchTensor,
        wTorchTensor, bTorchTensor, stridesList, padsList, dilationList,
        groupTorchInt);

    llvm::outs() << "AtenConv2d operation creation "
                 << "\n"
                 << result << "\n"
                 << "\n";

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);

    return success();
  }
};

void populateLoweringONNXToTorchConvOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConvOpToTorchLowering>(typeConverter, ctx);
}
