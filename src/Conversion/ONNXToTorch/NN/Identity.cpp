/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- SqrtOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2022, Helprack LLC.
//
// =============================================================================
//
// This file lowers most unary operators from torch to onnx using a template
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

struct ONNXToTorchIdentityOpLowering : public ConversionPattern {
  ONNXToTorchIdentityOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXIdentityOp::getOperationName(), 1, ctx) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto identityOp = llvm::dyn_cast_or_null<ONNXIdentityOp>(op);

    assert(identityOp && "Expecting op to have a strong type");

    Location loc = identityOp.getLoc();
    Value operand = identityOp.getOperand();
    mlir::MLIRContext *context =  identityOp.getContext();

    auto operandType = toTorchType(context, operand.getType());
    auto operandTensor = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, operandType, operand);

    llvm::outs() << "Input and output are "
                 << operandTensor
                 << "\n";

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, operandType, operandTensor);

    return success();
  }
};

void populateLoweringONNXToTorchIdentityOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXToTorchIdentityOpLowering>(typeConverter, ctx);
}
