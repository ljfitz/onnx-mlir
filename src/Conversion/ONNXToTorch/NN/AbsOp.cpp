/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- AbsOp.cpp - ONNX Op Transform ------------------===//
//
// ======================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

#ifdef _WIN32
#include <io.h>
#endif

/* ONNX Abs operation

 * “Absolute takes one input data (Tensor) and produces one output 
 * data" "(Tensor) where the absolute is, y = abs(x), is applied to" "the 
 * tensor elementwise."

 * Operands:
 * Operand Description
   * X	tensor of 8-bit/16-bit/32-bit/64-bit unsigned integer values or
   *    tensor of 8-bit/16-bit/32-bit/64-bit signless integer values or 
   *    tensor of 16-bit/32-bit/64-bit float values or 
   *    tensor of bfloat16 type values or memref of any type values.
   *  ONNX X is map to input in torch side.
 * Results:
 * Result Description
 * Y    tensor of 8-bit/16-bit/32-bit/64-bit unsigned integer values or
   *    tensor of 8-bit/16-bit/32-bit/64-bit signless integer values or
   *    tensor of 16-bit/32-bit/64-bit float values or
   *    tensor of bfloat16 type values or memref of any type values.
 */

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

class ONNXAbsOpToTorchLowering : public ConversionPattern {
public:
  ONNXAbsOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXAbsOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context =  op->getContext();
    ONNXAbsOp op1 = llvm::dyn_cast<ONNXAbsOp>(op);
    Value x = op1.X();

    TensorType xTensorType  = x.getType().cast<TensorType>();
    auto xType = Torch::ValueTensorType::get(context, 
		xTensorType.getShape(), xTensorType.getElementType());
    auto xTorchTensor  
	    = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
			    loc, xType, x);

    TensorType opTensorType =
            op->getResult(0).getType().cast<TensorType>();
    auto resultTy = Torch::ValueTensorType::get(op1.getContext(),
		opTensorType.getShape(), opTensorType.getElementType());

    Value result = rewriter.create<AtenAbsOp>(loc, resultTy, xTorchTensor); 
    llvm::outs() << "ATENABS CREATED is " << result << "\n" << "\n"; 
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTy,
		    result);
    return success();
  }
};

void populateLoweringONNXToTorchAbsOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXAbsOpToTorchLowering>(typeConverter, ctx);
}
