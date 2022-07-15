/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToTOSA.cpp - ONNX dialects to TOSA lowering -------===//
//
// Copyright (c) 2022 Arm Limited.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

using namespace mlir;

namespace onnx_mlir {

inline bool isa_tosa_signed_int(Type type) {
  IntegerType intType = type.dyn_cast<IntegerType>();
  std::set<unsigned> intWidth{8, 16, 32, 48, 64};
  return intType && (intWidth.find(intType.getWidth()) != intWidth.end());
}

inline bool isa_tosa_float(Type type) {
  return type.isa<BFloat16Type, Float16Type, Float32Type>();
}

void populateONNXToTOSAConversionPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  // Math
  populateLoweringONNXElementwiseOpToTOSAPattern(patterns, typeConverter, ctx);
  // Tensor
  populateLoweringONNXArgMaxOpToTOSAPattern(patterns, typeConverter, ctx);
}

// Performs lowering to TOSA dialect
struct FrontendToTosaLoweringPass
    : public PassWrapper<FrontendToTosaLoweringPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "convert-onnx-to-tosa"; }

  StringRef getDescription() const override {
    return "Lower frontend ops to TOSA dialect.";
  }

  FrontendToTosaLoweringPass() = default;
  FrontendToTosaLoweringPass(const FrontendToTosaLoweringPass &pass)
      : PassWrapper<FrontendToTosaLoweringPass, OperationPass<ModuleOp>>() {}

  void runOnOperation() final;
};

void FrontendToTosaLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();
  // Define final conversion target
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget target(*context);

  // We use the type converter to legalize types before any conversion patterns
  // are executed. This ensures that we do not need to trigger separate
  // conversion failures.
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) -> Optional<Type> {
    if (isa_tosa_signed_int(type) || isa_tosa_float(type))
      return type;
    return llvm::None;
  });
  typeConverter.addConversion([](TensorType type) -> Optional<Type> {
    if (isa_tosa_signed_int(type.getElementType()) ||
        isa_tosa_float(type.getElementType()))
      return type;
    return llvm::None;
  });

  // Define legal dialects and operations
  target.addLegalDialect<tosa::TosaDialect, func::FuncDialect>();

  // Define patterns
  populateONNXToTOSAConversionPattern(patterns, typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> createConvertONNXToTOSAPass() {
  return std::make_unique<FrontendToTosaLoweringPass>();
}

} // namespace onnx_mlir
