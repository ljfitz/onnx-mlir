/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- GlobalAveragePool.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2022, Helprack LLC.
//
// ========================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===-----------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/*
 * ONNX GlobalAveragePool operation
 *
 * “GlobalAveragePool consumes an input tensor X and applies average
 * pooling across” “ the values in the same channel.
 * This is equivalent to AveragePool with kernel size” “ equal to the
 * spatial dimension of input tensor.”
 *
 * Operands:
 *  X	tensor of 16-bit/32-bit/64-bit float values or memref of any
 *      type values
 * Results:
 *  Y	tensor of 16-bit/32-bit/64-bit float values or memref of any
 *      type values
 *
 */
struct ONNXGlobalAveragePoolOpToTorchLowering : public ConversionPattern {

  ONNXGlobalAveragePoolOpToTorchLowering(TypeConverter &typeConverter,
                                         MLIRContext *ctx)
      : ConversionPattern(typeConverter,
                          mlir::ONNXGlobalAveragePoolOp::getOperationName(), 1,
                          ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    ONNXGlobalAveragePoolOp globalAveragePool =
        llvm::dyn_cast_or_null<ONNXGlobalAveragePoolOp>(op);

    Value atenGlobAvgpool2d =
        rewriter.create<AtenAdaptiveAvgPool2dOp>(loc, resultTy, xtt, f1v);

    mlir::MLIRContext *context = globalAveragePool.getContext();
    Location loc = globalAveragePool.getLoc();

    auto x = globalAveragePool.X();
    auto resultType =
        toTorchType(context, globalAveragePool.getResult().getType());
    auto xTensor = getTorchTensor(x, rewriter, context, loc);

    Value one = getIntValue(1, rewriter, context, loc);

    Value hAndWDimensions = rewriter.create<PrimListConstructOp>(
        loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{one, one});

    Value result = rewriter.create<AtenAdaptiveAvgPool2dOp>(
        loc, resultType, xTensor, hAndWDimensions);

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);

    return success();
  }
};

void populateLoweringONNXToTorchGlobalAveragePoolOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXGlobalAveragePoolOpToTorchLowering>(typeConverter, ctx);
}
