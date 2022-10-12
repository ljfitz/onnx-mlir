/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

template <>
struct TOSADialectOp<ONNXNegOp> {
  using Op = tosa::NegateOp;
};

namespace {

// Element-wise unary ops lowering to TOSA dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOp>
class ONNXElementwiseUnaryOpLoweringToTOSA
    : public OpConversionPattern<ElementwiseUnaryOp> {
public:
  using OpConversionPattern<ElementwiseUnaryOp>::OpConversionPattern;
  using OpAdaptor = typename ElementwiseUnaryOp::Adaptor;
  LogicalResult matchAndRewrite(ElementwiseUnaryOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TOSAOp<ElementwiseUnaryOp>>(
        op, op.getType(), adaptor.X());
    return success();
  }
};

class ONNXFloorOpLoweringToTOSA : public OpConversionPattern<ONNXFloorOp> {
public:
  using OpConversionPattern<ONNXFloorOp>::OpConversionPattern;
  using OpAdaptor = typename ONNXFloorOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXFloorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto scalarType = getElementTypeOrSelf(adaptor.X());
    if (!isTOSAFloat(scalarType))
      return rewriter.notifyMatchFailure(
          op, "`tosa.floor` only supports float types");

    rewriter.replaceOpWithNewOp<tosa::FloorOp>(op, op.getType(), adaptor.X());
    return success();
  }
};

template <typename ONNXOpT, typename TosaOpT>
class ConvertONNXAddSubOpToTOSA : public OpConversionPattern<ONNXOpT> {
public:
  using OpConversionPattern<ONNXOpT>::OpConversionPattern;
  using OpAdaptor = typename ONNXOpT::Adaptor;
  LogicalResult matchAndRewrite(ONNXOpT op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value lhs = adaptor.A();
    auto lhsType = lhs.getType().dyn_cast<TensorType>();

    Value rhs = adaptor.B();
    auto rhsType = rhs.getType().dyn_cast<TensorType>();

    auto resultType = op.getResult().getType().template dyn_cast<TensorType>();
    if (!lhsType || !rhsType || !resultType) {
      return rewriter.notifyMatchFailure(op, "Tosa only supports TensorTypes");
    }

    Type resultElementType = resultType.getElementType();

    if (!resultElementType.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "Only Int and Float are supported");
    }

    rewriter.replaceOpWithNewOp<TosaOpT>(op, op.getType(), lhs, rhs);

    return success();
  }
};

class ONNXLeakyReluOpLoweringToTOSA
    : public OpConversionPattern<ONNXLeakyReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = ONNXLeakyReluOp::Adaptor;
  LogicalResult matchAndRewrite(ONNXLeakyReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    TensorType outputType = op.getResult().getType().dyn_cast<TensorType>();

    if (!outputType.getElementType().isF32()) {
      return rewriter.notifyMatchFailure(op, "Only float is supported");
    }

    FloatAttr alphaAttr = adaptor.alphaAttr();

    double alpha = 0.01;

    if (alphaAttr) {
      alpha = alphaAttr.getValueAsDouble();
    }

    Value constZero = tosa::getTosaConstTensorSingleF32(
        rewriter, op, 0.0, outputType.getShape());

    auto mul = tosa::CreateOpAndInfer<tosa::MulOp>(rewriter, op.getLoc(),
        outputType, adaptor.X(),
        tosa::getTosaConstTensorSingleF32(
            rewriter, op, alpha, outputType.getShape()),
        0);

    auto greaterEqual = tosa::CreateOpAndInfer<tosa::GreaterEqualOp>(rewriter,
        op.getLoc(), UnrankedTensorType::get(rewriter.getI1Type()), adaptor.X(),
        constZero);

    auto select = tosa::CreateOpAndInfer<tosa::SelectOp>(rewriter, op.getLoc(),
        outputType, greaterEqual, adaptor.X(), mul.getResult());

    rewriter.replaceOp(op, {select.getResult()});

    return success();
  }
};

class ONNXReluOpLoweringToTOSA : public OpConversionPattern<ONNXReluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Value input = adaptor.X();

    // Quantized types are not supported right now (in type conversion).
    // Once they are, the input should be rescaled for quantized types. (TBD)
    // Maps to `tosa.clamp` which has both int and fp limits.
    rewriter.replaceOpWithNewOp<tosa::ClampOp>(op, op.getType(), input,
        rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(std::numeric_limits<int32_t>::max()),
        rewriter.getF32FloatAttr(0.0f),
        rewriter.getF32FloatAttr(std::numeric_limits<float>::max()));
    return success();
  }
};

} // namespace

void populateLoweringONNXElementwiseOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXElementwiseUnaryOpLoweringToTOSA<ONNXNegOp>,
      ConvertONNXAddSubOpToTOSA<ONNXAddOp, tosa::AddOp>,
      ConvertONNXAddSubOpToTOSA<ONNXSubOp, tosa::SubOp>,
      ONNXFloorOpLoweringToTOSA, ONNXReluOpLoweringToTOSA,
      ONNXLeakyReluOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
