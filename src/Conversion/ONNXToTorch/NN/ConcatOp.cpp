/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ConcatOp.cpp - ONNX Op Transform -----------------------===//
//
// =======================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

#ifdef _WIN32
#include <io.h>
#endif

/*
 * “Concatenate a list of tensors into a single tensor. 
 * All input tensors must have the same shape, except for the dimension 
 * size of the axis to concatenate on.”
 *
 * Attributes:
 *	axis	::mlir::IntegerAttr	64-bit signed integer attribute
 *  ONNX axis value is map to dimension in the torch side.
 *
 * Operands:
 *    inputs	tensor of 8-bit/16-bit/32-bit/64-bit unsigned 
 *    		integer values or tensor of 8-bit/16-bit/32-bit/64-bit 
 *    		signless integer values or tensor of bfloat16 type values 
 *    		or tensor of 16-bit/32-bit/64-bit float values or 
 *    		tensor of string type values or tensor of 1-bit signless 
 *    		integer values or tensor of complex type with 32-bit/64-bit
 *    		float elements values or memref of any type values.
 *    ONNX inputs map to input tensors in torch side.
 *
 * Results:
 * concat_result    tensor of 8-bit/16-bit/32-bit/64-bit unsigned
 *              integer values or tensor of 8-bit/16-bit/32-bit/64-bit
 *              signless integer values or tensor of bfloat16 type values
 *              or tensor of 16-bit/32-bit/64-bit float values or
 *              tensor of string type values or tensor of 1-bit signless
 *              integer values or tensor of complex type with 32-bit/64-bit
 *              float elements values or memref of any type values.
 */
using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;


class ONNXConcatOpToTorchLowering : public ConversionPattern {
public:
  ONNXConcatOpToTorchLowering(TypeConverter &typeConverter, 
	MLIRContext *ctx)
      : ConversionPattern(
	typeConverter, ::mlir::ONNXConcatOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context =  op->getContext();
    ONNXConcatOp op1 = llvm::dyn_cast<ONNXConcatOp>(op);
    ONNXConcatOpAdaptor adaptor(op1);
    
    ValueRange inputs = op1.inputs();
    auto axisValue = op1.axisAttr();       // ::mlir::IntegerAttr
    Value  axisVal = rewriter.create<ConstantIntOp>(loc,axisValue);
    
    TensorType op_tensor_type = op1.getType().cast<TensorType>();
    auto resultTy = Torch::ValueTensorType::get(op1.getContext(),
	op_tensor_type.getShape(), op_tensor_type.getElementType());
    std::vector<Value> inputArrayValues;
    for (unsigned int i = 0; i < inputs.size(); i++)
    {
      TensorType inputTensorType =
	inputs[i].getType().cast<TensorType>();
      auto inputTy = Torch::ValueTensorType::get(context, 
	inputTensorType.getShape(), inputTensorType.getElementType());
      auto inputTorchTensor  = 
	rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
		      loc, inputTy, inputs[i]);
      inputArrayValues.push_back(inputTorchTensor);
    }
    Value inputShapeList = rewriter.create<PrimListConstructOp>(loc, 
	Torch::ListType::get(inputArrayValues.front().getType()), 
       			ValueRange{inputArrayValues}); 

    Value result = rewriter.create<AtenCatOp>(loc, resultTy, 
		    inputShapeList, axisVal);
    
    llvm::outs() << "Aten Concat Op:   " << "\n" << result 
	    << "\n" << "\n";
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, resultTy,
		    result);
    return success();
  }
};

void populateLoweringONNXToTorchConcatOpPattern(RewritePatternSet 
	&patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
    patterns.insert<ONNXConcatOpToTorchLowering>(typeConverter, ctx);
}
