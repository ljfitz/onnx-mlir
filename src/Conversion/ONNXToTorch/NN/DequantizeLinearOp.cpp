/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- DequantizeLinearOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2022, Helprack LLC.
//
// =============================================================================
//
// This file lowers DequantizeLinearOp from Onnx to Torch
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

struct ONNXDequantizeLinearOpToTorchLowering : public ConversionPattern {
  ONNXDequantizeLinearOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXDequantizeLinearOp::getOperationName(), 1, ctx) {}

  Value subOp(ConversionPatternRewriter &rewriter, Location loc, TensorValue a, Value b) {
    mlir::MLIRContext *context =  unaryOp.getContext();

    TensorType resultTensorType = unaryOp.getResult().getType().template dyn_cast<TensorType>();

    auto operandType = Torch::ValueTensorType::get(context,
                                           operandTensorType.getShape(),
                                           operandTensorType.getElementType());

    auto operandTensor = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, operandType, operand);

    auto resultType = Torch::ValueTensorType::get(context,
                                                resultTensorType.getShape(),
                                                resultTensorType.getElementType());

    llvm::outs() << "Unary input is "
                 << operandTensor
                 << "\n";

    Value result = rewriter.create<TorchUnaryOp>(loc, resultType, operandTensor);

    return result;
  }

  // y = (x - x_zero_point) * x_scale
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXDequantizeLinearOp dlOp = llvm::dyn_cast_or_null<ONNXDequantizeLinearOp>(op);

    assert(dlOp && "Expecting op to have a strong type");

    Location loc = dlOp.getLoc();
    mlir::MLIRContext *context =  dlOp.getContext();

    auto axis = dlOp.axisAttr(); // ::mlir::IntegerAttr

    auto x = dlOp.x();
    auto xScale = dlOp.x_scale();
    auto xZeroPoint = dlOp.x_zero_point();

    auto xType = x.getType().cast<TensorType>();
    auto xScaleType = xScale.getType().cast<TensorType>();
    auto xZeroPointType = xZeroPoint.getType().cast<TensorType>();

    auto xShape = x.getType().cast<ShapedType>().getShape();
    auto xScaleShape = xScale.getType().cast<ShapedType>().getShape();
    auto xZeroPointShape = xZeroPoint.getType().cast<ShapedType>().getShape();

    int64_t xRank = xShape.size();

    assert((xRank <= axis && xRank >= axis) && "Expecting axis to be within bounds");
    assert((xScaleShape == xZeroPointShape) && "Expecting shape of x_scale nad x_zero_point to be equal");



    return success();
  }
};

void populateLoweringONNXToTorchDequantizeLinearOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXDequantizeLinearOpToTorchLowering>(typeConverter, ctx);
}
