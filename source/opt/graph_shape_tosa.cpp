// Copyright (c) 2023-2025 Arm Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "source/opt/graph_shape_pass.h"
#include "source/opt/instruction.h"
#include "source/opt/types.h"

namespace tosa010 {
#include "spirv/unified1/TOSA.001000.1.h"
}

using namespace spvtools::opt::analysis;

namespace {

int32_t idiv_check(int32_t input1, int32_t input2) {
  assert(input1 % input2 == 0);
  return input1 / input2;
}

const std::map<uint32_t, std::vector<uint32_t> > TOSAOpInputsIdMap = {
    // Tensor Operators
    {tosa010::TOSAARGMAX, {2, 4}},
    {tosa010::TOSAAVG_POOL2D, {2, 3, 4, 6}},
    {tosa010::TOSAMAX_POOL2D, {2, 3, 4, 6}},
    {tosa010::TOSACONV2D, {2, 3, 4, 7, 8}},
    // Activation Operators
    {tosa010::TOSACLAMP, {5}},
    {tosa010::TOSAERF, {2}},
    {tosa010::TOSASIGMOID, {2}},
    {tosa010::TOSATANH, {2}},
    // Elementwise-binary operators
    {tosa010::TOSAADD, {2, 3}},
    {tosa010::TOSAARITHMETIC_RIGHT_SHIFT, {3, 4}},
    {tosa010::TOSABITWISE_AND, {2, 3}},
    {tosa010::TOSABITWISE_OR, {2, 3}},
    {tosa010::TOSABITWISE_XOR, {2, 3}},
    {tosa010::TOSAINTDIV, {2, 3}},
    {tosa010::TOSALOGICAL_AND, {2, 3}},
    {tosa010::TOSALOGICAL_LEFT_SHIFT, {2, 3}},
    {tosa010::TOSALOGICAL_RIGHT_SHIFT, {2, 3}},
    {tosa010::TOSALOGICAL_OR, {2, 3}},
    {tosa010::TOSALOGICAL_XOR, {2, 3}},
    {tosa010::TOSAMAXIMUM, {3, 4}},
    {tosa010::TOSAMINIMUM, {3, 4}},
    {tosa010::TOSAMUL, {2, 3}},
    {tosa010::TOSAPOW, {2, 3}},
    {tosa010::TOSASUB, {2, 3}},
    {tosa010::TOSATABLE, {2}},
    // Elementwise-unary operators
    {tosa010::TOSAABS, {2}},
    {tosa010::TOSABITWISE_NOT, {2}},
    {tosa010::TOSACEIL, {2}},
    {tosa010::TOSACLZ, {2}},
    {tosa010::TOSAEXP, {2}},
    {tosa010::TOSAFLOOR, {2}},
    {tosa010::TOSALOG, {2}},
    {tosa010::TOSALOGICAL_NOT, {2}},
    {tosa010::TOSANEGATE, {2}},
    {tosa010::TOSARECIPROCAL, {2}},
    {tosa010::TOSARSQRT, {2}},
    // Reduction operators
    {tosa010::TOSAREDUCE_ALL, {2, 3}},
    {tosa010::TOSAREDUCE_ANY, {2, 3}},
    {tosa010::TOSAREDUCE_MAX, {2, 4}},
    {tosa010::TOSAREDUCE_MIN, {2, 4}},
    {tosa010::TOSAREDUCE_PRODUCT, {2, 3}},
    {tosa010::TOSAREDUCE_SUM, {2, 3}},
    // Elementwise-ternary operators
    {tosa010::TOSASELECT, {2, 3, 4}},
    // Comparison operators
    {tosa010::TOSAEQUAL, {2, 3}},
    {tosa010::TOSAGREATER, {2, 3}},
    {tosa010::TOSAGREATER_EQUAL, {2, 3}},
    // Image Operators
    {tosa010::TOSARESIZE, {3, 4, 5, 6}},
    // Data-layout operators
    {tosa010::TOSACONCAT, {2, 3}},
};

}  // namespace

namespace spvtools {
namespace opt {

std::vector<int64_t> ShapeGraphInstructionArgmaxOp(IRContext* context,
                                                   uint32_t input_id,
                                                   uint32_t axis_id) {
  auto axis = context->get_constant_mgr()
                  ->FindDeclaredConstant(axis_id)
                  ->AsIntConstant()
                  ->GetU32();

  auto input_shape = *GetTensorShape(context, input_id);

  if (axis >= input_shape.size()) {
    assert(false && "Axis outside of input tensor rank");
  }

  std::vector<int64_t> output_shape;
  output_shape = input_shape;
  output_shape.erase(output_shape.begin() + axis);

  return output_shape;
}

std::vector<int64_t> ShapeGraphInstructionAvgMaxPool2dOps(IRContext* context,
                                                          uint32_t kernel_id,
                                                          uint32_t stride_id,
                                                          uint32_t pad_id,
                                                          uint32_t input_id) {
  auto kernel = GetConstantValuesOfKnownSizeI32(context, kernel_id, 2);
  int32_t kernel_y = kernel[0];
  int32_t kernel_x = kernel[1];

  auto stride = GetConstantValuesOfKnownSizeI32(context, stride_id, 2);
  int32_t stride_y = stride[0];
  int32_t stride_x = stride[1];

  auto pad = GetConstantValuesOfKnownSizeI32(context, pad_id, 4);
  int32_t pad_top = pad[0];
  int32_t pad_bottom = pad[1];
  int32_t pad_left = pad[2];
  int32_t pad_right = pad[3];

  auto input_shape = *GetTensorShape(context, input_id);
  assert(input_shape.size() == 4);

  auto N = input_shape[0];
  auto IH = static_cast<int32_t>(input_shape[1]);
  auto IW = static_cast<int32_t>(input_shape[2]);
  auto C = input_shape[3];

  int64_t OH = idiv_check(IH + pad_top + pad_bottom - kernel_y, stride_y) + 1;
  int64_t OW = idiv_check(IW + pad_left + pad_right - kernel_x, stride_x) + 1;

  return {N, OH, OW, C};
}

std::vector<int64_t> ShapeGraphInstructionMatchOutput(IRContext* context,
                                                      uint32_t input_id) {
  auto input_shape = *GetTensorShape(context, input_id);
  return input_shape;
}

std::vector<int64_t> ShapeGraphInstructionMatchOutputWithCheck(
    IRContext* context, uint32_t input1_id, uint32_t input2_id) {
  auto input1_shape = *GetTensorShape(context, input1_id);
  auto input2_shape = *GetTensorShape(context, input2_id);

  assert(input1_shape == input2_shape &&
         "Input shapes mismatch, which does not comply with the TOSA "
         "specification");

  return input1_shape;
}

std::vector<int64_t> ShapeGraphInstructionReduceOps(IRContext* context,
                                                    uint32_t input_id,
                                                    uint32_t axis_id) {
  auto axis = context->get_constant_mgr()
                  ->FindDeclaredConstant(axis_id)
                  ->AsIntConstant()
                  ->GetU32();

  auto input_shape = *GetTensorShape(context, input_id);

  if (axis >= input_shape.size()) {
    assert(false && "Axis outside of input tensor rank");
  }

  std::vector<int64_t> output_shape;
  output_shape = input_shape;
  output_shape[axis] = 1;
  return output_shape;
}

std::vector<int64_t> ShapeGraphInstructionSelectOp(IRContext* context,
                                                   uint32_t input2_id,
                                                   uint32_t input3_id) {
  auto input2_shape = *GetTensorShape(context, input2_id);

  auto input3_shape = *GetTensorShape(context, input3_id);

  assert(input2_shape == input3_shape &&
         "Input shapes mismatch, which does not comply with the TOSA "
         "specification");

  return input2_shape;
}

std::vector<int64_t> ShapeGraphInstructionConv2dOp(
    IRContext* context, uint32_t pad_id, uint32_t stride_id,
    uint32_t dilation_id, uint32_t input_id, uint32_t weights_id) {
  auto pad = GetConstantValuesOfKnownSizeI32(context, pad_id, 4);
  int32_t pad_top = pad[0];
  int32_t pad_bottom = pad[1];
  int32_t pad_left = pad[2];
  int32_t pad_right = pad[3];

  auto stride = GetConstantValuesOfKnownSizeI32(context, stride_id, 2);
  int32_t stride_y = stride[0];
  int32_t stride_x = stride[1];

  auto dilation = GetConstantValuesOfKnownSizeI32(context, dilation_id, 2);
  int32_t dilation_y = dilation[0];
  int32_t dilation_x = dilation[1];

  auto input_shape = *GetTensorShape(context, input_id);
  assert(input_shape.size() == 4);

  auto N = input_shape[0];
  auto IH = static_cast<int32_t>(input_shape[1]);
  auto IW = static_cast<int32_t>(input_shape[2]);

  auto weights_shape = *GetTensorShape(context, weights_id);
  assert(weights_shape.size() == 4);

  auto KO = weights_shape[0];
  auto KH = static_cast<int32_t>(weights_shape[1]);
  auto KW = static_cast<int32_t>(weights_shape[2]);

  int64_t OH = idiv_check(IH - 1 + pad_top + pad_bottom - (KH - 1) * dilation_y,
                          stride_y) +
               1;
  int64_t OW = idiv_check(IW - 1 + pad_left + pad_right - (KW - 1) * dilation_x,
                          stride_x) +
               1;

  return {N, OH, OW, KO};
}

std::vector<int64_t> ShapeGraphInstructionResizeOp(IRContext* context,
                                                   uint32_t input_id,
                                                   uint32_t scale_id,
                                                   uint32_t offset_id,
                                                   uint32_t border_id) {
  auto input_shape = *GetTensorShape(context, input_id);
  assert(input_shape.size() == 4);

  auto N = input_shape[0];
  auto IH = static_cast<int32_t>(input_shape[1]);
  auto IW = static_cast<int32_t>(input_shape[2]);
  auto IC = input_shape[3];

  if (context->get_constant_mgr()->FindDeclaredConstant(scale_id) == nullptr) {
    assert(false &&
           "Can't infer the output shape if the scale value is not known");
  }
  auto scale = GetConstantValuesOfKnownSizeI32(context, scale_id, 4);
  int32_t scale_y_n = scale[0];
  int32_t scale_y_d = scale[1];
  int32_t scale_x_n = scale[2];
  int32_t scale_x_d = scale[3];

  if (context->get_constant_mgr()->FindDeclaredConstant(offset_id) == nullptr) {
    assert(false &&
           "Can't infer the output shape if the offset value is not known");
  }
  auto offset = GetConstantValuesOfKnownSizeI32(context, offset_id, 2);
  int32_t offset_y = offset[0];
  int32_t offset_x = offset[1];

  if (context->get_constant_mgr()->FindDeclaredConstant(border_id) == nullptr) {
    assert(false &&
           "Can't infer the output shape if the border value is not known");
  }
  auto border = GetConstantValuesOfKnownSizeI32(context, border_id, 2);
  int32_t border_y = border[0];
  int32_t border_x = border[1];

  int64_t OH =
      idiv_check((IH - 1) * scale_y_n - offset_y + border_y, scale_y_d) + 1;
  int64_t OW =
      idiv_check((IW - 1) * scale_x_n - offset_x + border_x, scale_x_d) + 1;

  return {N, OH, OW, IC};
}

std::vector<int64_t> ShapeGraphInstructionConcatOp(IRContext* context,
                                                   Instruction* inst,
                                                   uint32_t axis_id,
                                                   uint32_t first_input_pos) {
  std::vector<int64_t> output_shape;
  auto axis = context->get_constant_mgr()
                  ->FindDeclaredConstant(axis_id)
                  ->AsIntConstant()
                  ->GetU32();

  auto num_operands = inst->NumInOperands();

  auto first_input_id = inst->GetSingleWordInOperand(first_input_pos);
  auto first_input_shape = *GetTensorShape(context, first_input_id);
  assert(first_input_shape.size() == 4);

  output_shape = first_input_shape;
  output_shape[axis] = 0;
  for (uint32_t i = 0; i < num_operands - 3; i++) {
    auto input_id = inst->GetSingleWordInOperand(i + first_input_pos);
    auto input_shape = *GetTensorShape(context, input_id);
    assert(input_shape.size() == 4);
    for (size_t dim = 0; dim < input_shape.size(); ++dim) {
      if (dim != axis) {
        assert(input_shape[dim] == first_input_shape[dim]);
      }
    }
    output_shape[axis] += input_shape[axis];
  }

  return output_shape;
}

Pass::Status ShapeGraphInstructionTOSA(IRContext* context,
                                                Instruction* inst) {
  auto tosaop = inst->GetSingleWordInOperand(1);

  std::vector<int64_t> output_shape;

  switch (tosaop) {
    case tosa010::TOSAARGMAX: {
      output_shape = ShapeGraphInstructionArgmaxOp(
          context,
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[1]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[0]));
      break;
    }
    case tosa010::TOSAAVG_POOL2D:
    case tosa010::TOSAMAX_POOL2D: {
      output_shape = ShapeGraphInstructionAvgMaxPool2dOps(
          context,
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[0]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[1]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[2]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[3]));
      break;
    }
    case tosa010::TOSAABS:
    case tosa010::TOSABITWISE_NOT:
    case tosa010::TOSACEIL:
    case tosa010::TOSACLZ:
    case tosa010::TOSAEXP:
    case tosa010::TOSAFLOOR:
    case tosa010::TOSALOG:
    case tosa010::TOSALOGICAL_NOT:
    case tosa010::TOSANEGATE:
    case tosa010::TOSARECIPROCAL:
    case tosa010::TOSARSQRT:
    case tosa010::TOSAERF:
    case tosa010::TOSASIGMOID:
    case tosa010::TOSATANH:
    case tosa010::TOSACLAMP:
    case tosa010::TOSATABLE: {
      output_shape = ShapeGraphInstructionMatchOutput(
          context,
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[0]));
      break;
    }
    case tosa010::TOSAADD:
    case tosa010::TOSAARITHMETIC_RIGHT_SHIFT:
    case tosa010::TOSABITWISE_AND:
    case tosa010::TOSABITWISE_OR:
    case tosa010::TOSABITWISE_XOR:
    case tosa010::TOSAINTDIV:
    case tosa010::TOSALOGICAL_AND:
    case tosa010::TOSALOGICAL_LEFT_SHIFT:
    case tosa010::TOSALOGICAL_RIGHT_SHIFT:
    case tosa010::TOSALOGICAL_OR:
    case tosa010::TOSALOGICAL_XOR:
    case tosa010::TOSAMAXIMUM:
    case tosa010::TOSAMINIMUM:
    case tosa010::TOSAMUL:
    case tosa010::TOSAPOW:
    case tosa010::TOSASUB:
    case tosa010::TOSAEQUAL:
    case tosa010::TOSAGREATER:
    case tosa010::TOSAGREATER_EQUAL: {
      output_shape = ShapeGraphInstructionMatchOutputWithCheck(
          context,
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[0]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[1]));
      break;
    }
    case tosa010::TOSAREDUCE_ALL:
    case tosa010::TOSAREDUCE_ANY:
    case tosa010::TOSAREDUCE_MAX:
    case tosa010::TOSAREDUCE_MIN:
    case tosa010::TOSAREDUCE_PRODUCT:
    case tosa010::TOSAREDUCE_SUM: {
      output_shape = ShapeGraphInstructionReduceOps(
          context,
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[1]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[0]));
      break;
    }
    case tosa010::TOSASELECT: {
      output_shape = ShapeGraphInstructionSelectOp(
          context,
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[1]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[2]));
      break;
    }
    case tosa010::TOSACONV2D: {
      output_shape = ShapeGraphInstructionConv2dOp(
          context,
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[0]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[1]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[2]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[3]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[4]));
      break;
    }
    case tosa010::TOSARESIZE: {
      output_shape = ShapeGraphInstructionResizeOp(
          context,
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[0]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[1]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[2]),
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[3]));
      break;
    }
    case tosa010::TOSACONCAT: {
      output_shape = ShapeGraphInstructionConcatOp(
          context, inst,
          inst->GetSingleWordInOperand(TOSAOpInputsIdMap.at(tosaop)[0]),
          TOSAOpInputsIdMap.at(tosaop)[1]);
      break;
    }
    case tosa010::TOSARESCALE: {
      // TOSA Rescale operator is a no-op in terms of shape inference. The input shape id is operand number 7.
      output_shape = ShapeGraphInstructionMatchOutput(
          context,
          inst->GetSingleWordInOperand(7));
      break;
    }
    default:
      assert(false && "Unhandled TOSA operator");
      break;
  }

  TensorARM* output_type =
      context->get_type_mgr()->GetType(inst->type_id())->AsTensorARM();
  auto output_shaped_tensor_type =
      GetShapedTensorType(context, output_type->element_type(), output_shape);
  uint32_t new_output_type_id =
      context->get_type_mgr()->GetTypeInstruction(output_shaped_tensor_type);
  uint32_t existing_output_type_id = inst->type_id();
  if (new_output_type_id == existing_output_type_id) {
    return Pass::Status::SuccessWithoutChange;
  } else {
    inst->SetResultType(new_output_type_id);
    return Pass::Status::SuccessWithChange;
  }
}

}  // namespace opt
}  // namespace spvtools
