/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Softmax.cpp - Softmax Op ---------------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX ArgMax operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXArgMaxOpLoweringToTOSA : public OpConversionPattern<ONNXArgMaxOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXArgMaxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    if (adaptor.keepdims() != 1)
      return rewriter.notifyMatchFailure(op, "keepdims != 1 is not supported");

    if (adaptor.select_last_index() != 0)
      return rewriter.notifyMatchFailure(
          op, "select_last_index != 0 is not supported");

    IntegerAttr axis = rewriter.getI64IntegerAttr(adaptor.axis());
    rewriter.replaceOpWithNewOp<tosa::ArgMaxOp>(
        op, op.getType(), adaptor.data(), axis);
    return success();
  }
};

} // namespace

void populateLoweringONNXArgMaxOpToTOSAPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXArgMaxOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
