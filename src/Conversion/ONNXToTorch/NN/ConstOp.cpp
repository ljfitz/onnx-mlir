/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ConstOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

#include <fstream>
#include <iostream>
#include <set>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/ToolOutputFile.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/OMOptions.hpp"

#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringExtras.h"

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/**
 * ONNX Constant  operation
 *
 * Creates the constant tensor.
 *
 * Operands : None.
 *
 * Validation
 * ----------
 * /scripts/docker/build_with_docker.py --external-build --build-dir build
 * --command
 * "build/Ubuntu1804-Release/third-party/onnx-mlir/Release/bin/onnx-mlir
 * --EmitONNXIR --debug --run-torch-pass
 * third-party/onnx-mlir/third_party/onnx/onnx/backend/test/data/node/
 * test_constant/model.onnx"
 *
 * Limitations
 * -----------
 * uses literal.
 *
 * TODO: Not handling String attribute in the ConstOp.
 */
class ONNXConstOpToTorchLowering : public ConversionPattern {
public:
  ONNXConstOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
          typeConverter, mlir::ONNXConstantOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    mlir::MLIRContext *context = op->getContext();
    ONNXConstantOp op1 = llvm::dyn_cast<ONNXConstantOp>(op);

    auto value_attr = op1.valueAttr(); // ::mlir::Attribute

    // Steps
    // 1) Extract float attributes array from ONNX and compare with 
    //      the Netron file,
    // 2) Find the shape of this array in step 1,
    // 3) Create the result type,
    // 4) Create the torch tensor of shape as in 2,
    // 5) Create the torch op and replace it.

    llvm::outs() << "CONSTFLOATOP operation creation value_attr type: "
                 << value_attr.getType() << "\n"
                 << "\n";
    llvm::outs() << "CONSTFLOATOP array tensor type 1: " << value_attr << "\n"
                 << "\n";

    TensorType op_tensor_type = op->getResult(0).getType().cast<TensorType>();
    ::mlir::Attribute value_attr_finalized;
    Type element_type;
    if (op_tensor_type) {
      if (auto integerType =
              op_tensor_type.getElementType().dyn_cast<IntegerType>()) {
        element_type = IntegerType::get(
            context, integerType.getWidth(), IntegerType::Signed);
        auto dense_value_attr =
            value_attr.dyn_cast<::mlir::DenseElementsAttr>();
        ShapedType dense_value_type =
            RankedTensorType::get(op_tensor_type.getShape(), element_type);
        std::vector<APInt> intValues;
        for (auto n : dense_value_attr.getValues<APInt>())
          intValues.push_back(n);
        auto new_dense_value_attr =
            DenseElementsAttr::get(dense_value_type, intValues);
        value_attr_finalized = new_dense_value_attr;
      } else if (auto floatType = op_tensor_type.getElementType()
                                      .dyn_cast<::mlir::FloatType>()) {
        element_type = ::mlir::FloatType::getF32(context);
        auto dense_value_attr =
            value_attr.dyn_cast<::mlir::DenseElementsAttr>();
        ShapedType dense_value_type =
            RankedTensorType::get(op_tensor_type.getShape(), element_type);
        std::vector<APFloat> floatValues;
        for (auto n : dense_value_attr.getValues<APFloat>())
          floatValues.push_back(n);
        auto new_dense_value_attr =
            DenseElementsAttr::get(dense_value_type, floatValues);
        value_attr_finalized = new_dense_value_attr;
      } /* else if (auto stringType = op_tensor_type.getElementType()
                                        .dyn_cast<::mlir::StringType>()) {
         element_type = ::mlir::StringType::get(context);
         auto dense_value_attr =
             value_attr.dyn_cast<::mlir::DenseElementsAttr>();
         ShapedType dense_value_type =
             RankedTensorType::get(op_tensor_type.getShape(), element_type);
         std::vector<StringRef> stringValues;
         for (auto n : dense_value_attr.getValues<StringRef>())
           stringValues.push_back(n);
         auto new_dense_value_attr =
             DenseElementsAttr::get(dense_value_type, stringValues);
         value_attr_finalized = new_dense_value_attr;
         } */
      else {
        element_type = op_tensor_type.getElementType();
        value_attr_finalized = value_attr;
      }
    }
    // for scalar types(int, float and string types).
    else {
      if (auto intType = value_attr.getType().cast<IntegerType>()) {
        element_type = ::mlir::IntegerType::get(
            context, intType.getWidth(), IntegerType::Signed);
        value_attr_finalized = value_attr;
      } else if (value_attr.getType().cast<::mlir::FloatType>()) {
        element_type = ::mlir::FloatType::getF32(context);
        value_attr_finalized = value_attr;
      } // else if (value_attr.getType().cast<::mlir::StringAttr>()) {
        // element_type = ::mlir::StringAttr::get(context);
        // value_attr_finalized = value_attr;
      //}
    }
    auto resultTy = Torch::ValueTensorType::get(
        op1.getContext(), op_tensor_type.getShape(), element_type);

    llvm::outs() << "CONSTFLOATOP operation creation: result type "
                 << "\n"
                 << resultTy << "\n"
                 << "\n";

    Value literal = rewriter.create<Torch::ValueTensorLiteralOp>(
        loc, resultTy, value_attr_finalized);

    llvm::outs() << "ValueTensorLiteralOp operation creation"
                 << "\n"
                 << literal << "\n"
                 << "\n";

    Value result = literal;

    llvm::outs() << "Before Writer replace Op"
                 << "\n";

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);

    llvm::outs() << "After Writer replace Op"
                 << "\n";

    return success();
  }
};

void populateLoweringONNXToTorchConstOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConstOpToTorchLowering>(typeConverter, ctx);
}