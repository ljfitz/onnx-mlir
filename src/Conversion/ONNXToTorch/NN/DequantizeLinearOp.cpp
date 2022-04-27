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

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

struct ONNXDequantizeLinearOpToTorchLowering : public ConversionPattern {
  ONNXDequantizeLinearOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXDequantizeLinearOp::getOperationName(), 1, ctx) {}

  Value subOp(
    ConversionPatternRewriter &rewriter,
    Location loc,
    mlir::MLIRContext *context,
    Torch::ValueTensorType resultType,
    Value a,
    Value b)
    const
  {
    auto aType = toTorchType(context, a.getType());
    auto bType = toTorchType(context, b.getType());

    auto aTensor = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(loc, aType, a);
    auto bTensor = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(loc, bType, b);

    auto one = 1;
    auto ty = IntegerType::get(context, 64);
    auto oneAttr = IntegerAttr::get(ty, one);
    Value alpha = rewriter.create<ConstantIntOp>(loc, oneAttr);

    return rewriter.create<AtenSubTensorOp>(loc, resultType, aTensor, bTensor, alpha);
  }

  Value mulOp(
    ConversionPatternRewriter &rewriter,
    Location loc,
    mlir::MLIRContext *context,
    Torch::ValueTensorType resultType,
    Value a,
    Value b)
    const
  {
    auto aType = toTorchType(context, a.getType());
    auto bType = toTorchType(context, b.getType());

    auto aTensor = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(loc, aType, a);
    auto bTensor = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(loc, bType, b);

    return rewriter.create<AtenMulTensorOp>(loc, resultType, aTensor, bTensor);
  }

  bool isShapeEqual(Value x, Value y) const {
    auto xShape = x.getType().cast<ShapedType>().getShape();
    auto yShape = y.getType().cast<ShapedType>().getShape();

    return xShape == yShape;
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

    auto xRank = x.getType().cast<ShapedType>().getShape().size();

    // assert((xRank <= axis && xRank >= axis) && "Expecting axis to be within bounds");
    // assert(isShapeEqual(xScale, xZeroPoint) && "Expecting shape of x_scale nad x_zero_point to be equal");

    auto resultType = toTorchType(context, dlOp.getResult().getType());
    auto result = mulOp(rewriter, loc, context, resultType,
                        subOp(rewriter, loc, context, resultType,
                              x,
                              xZeroPoint),
                        xScale);
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultType, result);

    return success();
  }
};

void populateLoweringONNXToTorchDequantizeLinearOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXDequantizeLinearOpToTorchLowering>(typeConverter, ctx);
}
