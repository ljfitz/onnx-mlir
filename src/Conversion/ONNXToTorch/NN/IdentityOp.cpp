/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- IdentityOp.cpp - ONNX Op Transform ------------------===//
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

/**
 * Performs identity operation. Take an input and returns the same thing as
 * output.
 *
 * Operands:
 *    input:     tensor of 8-bit unsigned integer values or tensor of 16-bit
 * unsigned integer values or tensor of 32-bit unsigned integer values or tensor
 * of 64-bit unsigned integer values or tensor of 8-bit signless integer values
 * or tensor of 16-bit signless integer values or tensor of 32-bit signless
 * integer values or tensor of 64-bit signless integer values or tensor of
 * bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit
 * float values or tensor of 64-bit float values or tensor of string type values
 * or tensor of 1-bit signless integer values or tensor of complex type with
 * 32-bit float elements values or tensor of complex type with 64-bit float
 * elements values or memref of any type values
 *
 * Results:
 *    output:    tensor of 8-bit unsigned integer values or tensor of 16-bit
 * unsigned integer values or tensor of 32-bit unsigned integer values or tensor
 * of 64-bit unsigned integer values or tensor of 8-bit signless integer values
 * or tensor of 16-bit signless integer values or tensor of 32-bit signless
 * integer values or tensor of 64-bit signless integer values or tensor of
 * bfloat16 type values or tensor of 16-bit float values or tensor of 32-bit
 * float values or tensor of 64-bit float values or tensor of string type values
 * or tensor of 1-bit signless integer values or tensor of complex type with
 * 32-bit float elements values or tensor of complex type with 64-bit float
 * elements values or memref of any type values
 */

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
    mlir::MLIRContext *context = identityOp.getContext();

    auto operandType = toTorchType(context, operand.getType());
    auto operandTensor =
        rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
            loc, operandType, operand);

    llvm::outs() << "Input and output are " << operandTensor << "\n";

    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(
        op, operandType, operandTensor);

    return success();
  }
};

void populateLoweringONNXToTorchIdentityOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXToTorchIdentityOpLowering>(typeConverter, ctx);
}
