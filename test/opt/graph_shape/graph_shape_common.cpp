// Copyright (c) 2024 Arm Ltd.
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
#include "graph_shape_common.h"

#include <regex>

namespace {

const std::regex opNameRgx("\\{op_name\\}");

}  // namespace

namespace spvtools {
namespace opt {

auto genShader = [](std::vector<std::string> insElemTypes,
                    std::vector<std::string> outsElemTypes,
                    MapOfInterfaceTensorShapes insShapes,
                    MapOfInterfaceTensorShapes outsShapes,
                    std::vector<std::pair<std::string, std::string>> attributes,
                    std::vector<std::pair<std::string, std::string>> postInputAttributes,
                    std::string extras, std::string tosaVersion, bool knownOutputShapes) {
  auto numOfInputs = insElemTypes.size();
  auto numOfOutputs = outsElemTypes.size();

  std::string shader = header;
  shader += "\n";
  shader += "%1 = OpExtInstImport \"" + tosaVersion + "\"";

  uint32_t bindingId = 0;
  for (uint32_t i = 0; i < numOfInputs; ++i) {
    auto str_i = std::to_string(i);
    auto strBindingId = std::to_string(bindingId);
    auto strInputSSA = "%{op_name}_arg_" + str_i;
    shader += "\n";
    shader += "OpDecorate " + strInputSSA + " Binding " + strBindingId;
    shader += "\n";
    shader += "OpDecorate " + strInputSSA + " DescriptorSet 0";
    bindingId++;
  }
  for (uint32_t i = 0; i < numOfOutputs; ++i) {
    auto str_i = std::to_string(i);
    auto strBindingId = std::to_string(bindingId);
    auto strOutputSSA = "%{op_name}_res_" + str_i;
    shader += "\n";
    shader += "OpDecorate " + strOutputSSA + " Binding " + strBindingId;
    shader += "\n";
    shader += "OpDecorate " + strOutputSSA + " DescriptorSet 0";
    bindingId++;
  }

  shader += definitions;

  if (extras != "") {
    shader += "\n";
    shader += extras;
  }

  if (!attributes.empty()) {
    for (auto& [attr, val] : attributes) {
      shader += "\n";
      shader += "%" + attr + " = " + val;
    }
  }
  if (!postInputAttributes.empty()) {
    for (auto& [postInputattr, val] : postInputAttributes) {
      shader += "\n";
      shader += "%" + postInputattr + " = " + val;
    }
  }

  std::set<std::string> spirvTypes;
  std::map<std::string, std::pair<std::string, std::string>> mapTypesAndSSA;
  std::map<uint32_t, std::pair<std::string, std::string>>
      mapInputBindingIdAndSSA;
  std::map<uint32_t, std::pair<std::string, std::string>>
      mapOutputBindingIdAndSSA;
  bindingId = 0;
  uint32_t id = 0;
  for (; bindingId < numOfInputs; ++bindingId) {
    auto strInputRank = std::to_string(insShapes.at({0, bindingId}).size());

    auto strInputType = "OpTypeTensorARM %" + insElemTypes[bindingId] +
                        " %uint_" + strInputRank;

    auto it = mapTypesAndSSA.find(strInputType);
    if (it != mapTypesAndSSA.end()) {
      // Type is already present
      mapInputBindingIdAndSSA.insert({bindingId, it->second});
      continue;
    }

    auto strId = std::to_string(id);
    auto strInputTypeSSA = "%input" + strId + "_type";
    auto strInputPtrType = "OpTypePointer UniformConstant " + strInputTypeSSA;
    auto strInputPtrTypeSSA = "%_ptr_UniformConstant_input" + strId;

    mapTypesAndSSA.insert(
        {strInputType, {strInputTypeSSA, strInputPtrTypeSSA}});
    mapInputBindingIdAndSSA.insert(
        {bindingId, {strInputTypeSSA, strInputPtrTypeSSA}});

    shader += "\n";
    shader += strInputTypeSSA + " = " + strInputType;
    shader += "\n";
    shader += strInputPtrTypeSSA + " = " + strInputPtrType;

    ++id;
  }

  id = 0;
  for (; bindingId < numOfInputs + numOfOutputs; ++bindingId) {
    auto strOutputRank = std::to_string(outsShapes.at({0, bindingId}).size());
    auto strId = std::to_string(id);
    std::string strOutputShape = "";
    std::string strOutputShapeSSA = "";
    if (knownOutputShapes) {
      strOutputShapeSSA = "%output" + strId + "_shape";
      strOutputShape = "OpConstantComposite %_arr_uint_uint_" + strOutputRank;
      for (auto dim : outsShapes.at({0, bindingId})) {
        strOutputShape += " %uint_" + std::to_string(dim);
      }
      shader += "\n";
      shader += strOutputShapeSSA + " = " + strOutputShape;
    }
    auto strOutputType =
        "OpTypeTensorARM %" + outsElemTypes[id] + " %uint_" + strOutputRank;
    if (!strOutputShapeSSA.empty()) {
      strOutputType += " " + strOutputShapeSSA;
    }

    auto it = mapTypesAndSSA.find(strOutputType);
    if (it != mapTypesAndSSA.end()) {
      // Type is already present
      mapOutputBindingIdAndSSA.insert({bindingId, it->second});
      continue;
    }

    auto strOutputTypeSSA = "%output" + strId + "_type";
    auto strOutputPtrType = "OpTypePointer UniformConstant " + strOutputTypeSSA;
    auto strOutputPtrTypeSSA = "%_ptr_UniformConstant_output" + strId;

    mapTypesAndSSA.insert(
        {strOutputType, {strOutputTypeSSA, strOutputPtrTypeSSA}});
    mapOutputBindingIdAndSSA.insert(
        {bindingId, {strOutputTypeSSA, strOutputPtrTypeSSA}});

    shader += "\n";
    shader += strOutputTypeSSA + " = " + strOutputType;
    shader += "\n";
    shader += strOutputPtrTypeSSA + " = " + strOutputPtrType;

    ++id;
  }
  bindingId = 0;
  std::vector<std::string> inputSSAs;
  for (uint32_t i = 0; i < numOfInputs; ++i) {
    auto str_i = std::to_string(i);
    auto strInputSSA = "%{op_name}_arg_" + str_i;
    shader += "\n";
    shader += strInputSSA + " = OpVariable " +
              mapInputBindingIdAndSSA[bindingId].second + " UniformConstant";
    inputSSAs.push_back(strInputSSA);
    bindingId++;
  }
  std::vector<std::string> outputSSAs;
  for (uint32_t i = 0; i < numOfOutputs; ++i) {
    auto str_i = std::to_string(i);
    auto strOutputSSA = "%{op_name}_res_" + str_i;
    shader += "\n";
    shader += strOutputSSA + " = OpVariable " +
              mapOutputBindingIdAndSSA[bindingId].second + " UniformConstant";
    outputSSAs.push_back(strOutputSSA);
    bindingId++;
  }

  shader += "\n";
  shader += "%graph_type = OpTypeGraphARM " + std::to_string(numOfInputs);
  for (auto& [_, ssa] : mapInputBindingIdAndSSA) {
    shader += " " + ssa.first;
  }
  for (auto& [_, ssa] : mapOutputBindingIdAndSSA) {
    shader += " " + ssa.first;
  }

  shader += "\n";
  shader += "OpGraphEntryPointARM %graph \"{op_name}\"";
  for (auto& ssa : inputSSAs) {
    shader += " " + ssa;
  }
  for (auto& ssa : outputSSAs) {
    shader += " " + ssa;
  }

  shader += "\n";
  shader += "%graph = OpGraphARM %graph_type";

  id = 0;
  std::vector<std::string> inputGraphSSAs;
  for (auto& [_, ssa] : mapInputBindingIdAndSSA) {
    shader += "\n";
    auto strGraphInputSSA = "%graph_input" + std::to_string(id);
    inputGraphSSAs.push_back(strGraphInputSSA);
    shader += strGraphInputSSA + " = OpGraphInputARM " + ssa.first + " %uint_" + std::to_string(id);
    id++;
  }

  // TODO: Needs correction for FFT2D and RFFT2D
  shader += "\n";
  shader += "%graph_output = OpExtInst";
  for (auto& [_, ssa] : mapOutputBindingIdAndSSA) {
    shader += " " + ssa.first;
  }
  shader += " %1 {op_name}";
  if (!attributes.empty()) {
    for (auto& [attr, _] : attributes) {
      shader += " %" + attr;
    }
  }

  for (auto& ssa : inputGraphSSAs) {
    shader += " " + ssa;
  }

  if (!postInputAttributes.empty()) {
    for (auto& [preInputAttr, _] : postInputAttributes) {
      shader += " %" + preInputAttr;
    }
  }

  shader += "\n";
  shader += "OpGraphSetOutputARM %graph_output %uint_0";
  shader += "\n";
  shader += "OpGraphEndARM";

  return shader;
};

TEST_P(GraphShapeSingleOpTestPass, Case) {
  auto shader = genShader(GetParam().insElemTypes, GetParam().outsElemTypes,
                          GetParam().insShapes, GetParam().outsShapes,
                          GetParam().attributes, GetParam().postInputAttributes,
                          GetParam().extras, GetParam().tosaVersion, false);

  auto numOfInputs = GetParam().insElemTypes.size();
  auto numOfOutputs = GetParam().outsElemTypes.size();

  // Add checks
  uint32_t bindingId = 0;
  uint32_t id = 0;
  std::vector<std::string> inputSSAs;
  std::map<std::string, std::string> mapShapesAndSSA;
  std::map<std::string, std::pair<std::string, std::string>> mapTypesAndSSA;
  std::map<uint32_t, std::pair<std::string, std::string>>
      mapInputBindingIdAndSSA;
  std::map<uint32_t, std::pair<std::string, std::string>>
      mapOutputBindingIdAndSSA;
  for (; bindingId < numOfInputs; ++bindingId) {
    auto strInputRank =
        std::to_string(GetParam().insShapes.at({0, bindingId}).size());

    auto strId = std::to_string(id);
    std::string strInputShapeSSA = "input" + strId + "_shape";
    std::string strInputShape =
        "OpConstantComposite %_arr_uint_uint_" + strInputRank;
    for (auto dim : GetParam().insShapes.at({0, bindingId})) {
      strInputShape += " %uint_" + std::to_string(dim);
    }

    auto strInputType = "OpTypeTensorARM %" +
                        GetParam().insElemTypes[bindingId] + " %uint_" +
                        strInputRank;

    auto itShapes = mapShapesAndSSA.find(strInputShape);
    if (itShapes == mapShapesAndSSA.end()) {
      mapShapesAndSSA[strInputShape] = strInputShapeSSA;
      shader += "\n; CHECK: ";
      shader += "[[" + strInputShapeSSA + ":%\\w+]] = " + strInputShape;
      strInputType += " [[" + strInputShapeSSA + "]]";
    } else {
      strInputType += " [[" + itShapes->second + "]]";
    }

    auto str_i = std::to_string(id);
    auto strInputSSA = "var_tensor_in" + str_i;
    inputSSAs.push_back(strInputSSA);

    auto itTypes = mapTypesAndSSA.find(strInputType);
    if (itTypes != mapTypesAndSSA.end()) {
      // Type is already present
      mapInputBindingIdAndSSA.insert({bindingId, itTypes->second});

      shader += "\n; CHECK: ";
      shader += "[[" + strInputSSA + ":%\\w+]] = OpVariable [[" +
                mapInputBindingIdAndSSA[bindingId].second +
                "]] UniformConstant";

      continue;
    }

    auto strInputTypeSSA = "input" + strId + "_type";
    auto strInputPtrType =
        "OpTypePointer UniformConstant [[" + strInputTypeSSA + "]]";
    auto strInputPtrTypeSSA = "_ptr_UniformConstant_input" + strId;

    mapTypesAndSSA.insert(
        {strInputType, {strInputTypeSSA, strInputPtrTypeSSA}});
    mapInputBindingIdAndSSA.insert(
        {bindingId, {strInputTypeSSA, strInputPtrTypeSSA}});
    shader += "\n; CHECK: ";
    shader += "[[" + strInputTypeSSA + ":%\\w+]] = " + strInputType;

    shader += "\n; CHECK: ";
    shader += "[[" + strInputPtrTypeSSA + ":%\\w+]] = " + strInputPtrType;

    shader += "\n; CHECK: ";
    shader += "[[" + strInputSSA + ":%\\w+]] = OpVariable [[" +
              mapInputBindingIdAndSSA[bindingId].second + "]] UniformConstant";

    ++id;
  }

  id = 0;
  bindingId = static_cast<uint32_t>(numOfInputs);
  std::vector<std::string> outputSSAs;
  for (; bindingId < numOfInputs + numOfOutputs; ++bindingId) {
    auto strOutputRank =
        std::to_string(GetParam().outsShapes.at({0, bindingId}).size());
    auto strId = std::to_string(id);
    auto strOutputShapeSSA = "output" + strId + "_shape";
    auto strOutputShape =
        "OpConstantComposite %_arr_uint_uint_" + strOutputRank;
    for (auto dim : GetParam().outsShapes.at({0, bindingId})) {
      strOutputShape += " %uint_" + std::to_string(dim);
    }

    auto strOutputType = "OpTypeTensorARM %" + GetParam().outsElemTypes[id] +
                         " %uint_" + strOutputRank;

    auto itShapes = mapShapesAndSSA.find(strOutputShape);
    if (itShapes == mapShapesAndSSA.end()) {
      mapShapesAndSSA[strOutputShape] = strOutputShapeSSA;
      shader += "\n; CHECK: ";
      shader += "[[" + strOutputShapeSSA + ":%\\w+]] = " + strOutputShape;
      strOutputType += " [[" + strOutputShapeSSA + "]]";
    } else {
      strOutputType += " [[" + itShapes->second + "]]";
    }

    auto str_i = std::to_string(id);
    auto strOutputSSA = "var_tensor_out" + str_i;
    outputSSAs.push_back(strOutputSSA);

    auto itTypes = mapTypesAndSSA.find(strOutputType);
    if (itTypes != mapTypesAndSSA.end()) {
      // Type is already present
      mapOutputBindingIdAndSSA.insert({bindingId, itTypes->second});

      shader += "\n; CHECK-DAG: ";
      shader += "[[" + strOutputSSA + ":%\\w+]] = OpVariable [[" +
                mapOutputBindingIdAndSSA[bindingId].second +
                "]] UniformConstant";

      continue;
    }

    auto strOutputTypeSSA = "output" + strId + "_type";
    auto strOutputPtrType =
        "OpTypePointer UniformConstant [[" + strOutputTypeSSA + "]]";
    auto strOutputPtrTypeSSA = "_ptr_UniformConstant_output" + strId;

    mapTypesAndSSA.insert(
        {strOutputType, {strOutputTypeSSA, strOutputPtrTypeSSA}});
    mapOutputBindingIdAndSSA.insert(
        {bindingId, {strOutputTypeSSA, strOutputPtrTypeSSA}});

    shader += "\n; CHECK: ";
    shader += "[[" + strOutputTypeSSA + ":%\\w+]] = " + strOutputType;

    shader += "\n; CHECK-DAG: ";
    shader += "[[" + strOutputPtrTypeSSA + ":%\\w+]] = " + strOutputPtrType;

    shader += "\n; CHECK-DAG: ";
    shader += "[[" + strOutputSSA + ":%\\w+]] = OpVariable [[" +
              mapOutputBindingIdAndSSA[bindingId].second + "]] UniformConstant";

    ++id;
  }

  shader += "\n; CHECK-DAG: ";
  shader +=
      "[[graph_type:%\\w+]] = OpTypeGraphARM " + std::to_string(numOfInputs);
  for (auto& [_, ssa] : mapInputBindingIdAndSSA) {
    shader += " [[" + ssa.first + "]]";
  }
  for (auto& [_, ssa] : mapOutputBindingIdAndSSA) {
    shader += " [[" + ssa.first + "]]";
  }

  shader += "\n; CHECK: ";
  shader += "OpGraphEntryPointARM [[graph:%\\w+]] \"{op_name}\"";
  for (auto& ssa : inputSSAs) {
    shader += " [[" + ssa + "]]";
  }
  for (auto& ssa : outputSSAs) {
    shader += " [[" + ssa + "]]";
  }

  shader += "\n; CHECK: ";
  shader += "[[graph]] = OpGraphARM [[graph_type]]";

  id = 0;
  std::vector<std::string> inputGraphSSAs;
  for (auto& [_, ssa] : mapInputBindingIdAndSSA) {
    shader += "\n; CHECK: ";
    auto strGraphInputSSA = "graph_input" + std::to_string(id);
    inputGraphSSAs.push_back(strGraphInputSSA);
    shader += "[[" + strGraphInputSSA + ":%\\w+]] = OpGraphInputARM [[" +
              ssa.first + "]]";
    id++;
  }

  // TODO: Needs correction for FFT2D and RFFT2D
  shader += "\n; CHECK: ";
  shader += "[[graph_output:%\\w+]] = OpExtInst";
  for (auto& [_, ssa] : mapOutputBindingIdAndSSA) {
    shader += " [[" + ssa.first + "]]";
  }
  shader += " %1 {op_name}";
  if (!GetParam().attributes.empty()) {
    shader += " {{.*}}";
  }

  for (auto& ssa : inputGraphSSAs) {
    shader += " [[" + ssa + "]]";
  }

  shader += "\n; CHECK: ";
  shader += "OpGraphSetOutputARM [[graph_output]]";
  shader += "\n; CHECK: ";
  shader += "OpGraphEndARM";

  shader = std::regex_replace(shader, opNameRgx, GetParam().OpNameUpperCase);

  auto result = SinglePassRunAndMatch<GraphShapePass>(
      shader, /* do_validate = */ true, GetParam().insShapes);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_P(GraphShapeSingleOpTestFail, Case) {
  auto shader = genShader(GetParam().insElemTypes, GetParam().outsElemTypes,
                          GetParam().insShapes, GetParam().outsShapes,
                          GetParam().attributes, GetParam().postInputAttributes,
                          GetParam().extras, GetParam().tosaVersion, true);

  // Add checks
  shader += "\n; CHECK: ";
  shader += "Tensor already shaped with a shape different from that requested";
  shader = std::regex_replace(shader, opNameRgx, GetParam().OpNameUpperCase);

  SinglePassRunAndFail<GraphShapePass>(shader, GetParam().insShapes);
}


using GraphShapeMultiOpTestPass = PassTest<::testing::Test>;
TEST_F(GraphShapeMultiOpTestPass, MaxPool2d_Conv2d) {
  std::string shader = header;
  shader += "\n";
  shader += "%1 = OpExtInstImport \"TOSA.001000.1\"";
  shader += R"(
    OpDecorate %arg_input Binding 0
    OpDecorate %arg_input DescriptorSet 0
    OpDecorate %arg_weight Binding 1
    OpDecorate %arg_weight DescriptorSet 0
    OpDecorate %arg_bias Binding 2
    OpDecorate %arg_bias DescriptorSet 0
    OpDecorate %arg_output Binding 3
    OpDecorate %arg_output DescriptorSet 0
  )";

  shader += definitions;

  shader += R"(
    %shape1 = OpConstantComposite %_arr_uint_uint_1 %uint_1
    %zero_point_type = OpTypeTensorARM %uchar %uint_1 %shape1
    %shape2 = OpConstantComposite %_arr_uint_uint_1 %uint_2
    %shape4 = OpConstantComposite %_arr_uint_uint_1 %uint_4
    %tensor_shape2 = OpTypeTensorARM %uint %uint_1 %shape2
    %tensor_shape4 = OpTypeTensorARM %uint %uint_1 %shape4
    %kernel = OpConstantComposite %tensor_shape2 %uint_2 %uint_2
    %stride = OpConstantComposite %tensor_shape2 %uint_2 %uint_2
    %pad    = OpConstantComposite %tensor_shape4 %uint_0 %uint_0 %uint_0 %uint_0
    %dilation = OpConstantComposite %tensor_shape2 %uint_1 %uint_1
    %acc_type = OpConstant %uint 1
    %nan_mode = OpConstant %uint 1
    %input_zero_point = OpConstantNull %zero_point_type
    %weight_zero_point = OpConstantNull %zero_point_type
    %local_bound = OpConstantFalse %bool
    %input_type      = OpTypeTensorARM %uchar %uint_4
    %bias_type       = OpTypeTensorARM %uint %uint_1
    %output_type     = OpTypeTensorARM %uint %uint_4
    %_ptr_input  = OpTypePointer UniformConstant %input_type
    %_ptr_weight = OpTypePointer UniformConstant %input_type
    %_ptr_bias   = OpTypePointer UniformConstant %bias_type
    %_ptr_output = OpTypePointer UniformConstant %output_type
    %arg_input   = OpVariable %_ptr_input UniformConstant
    %arg_weight  = OpVariable %_ptr_weight UniformConstant
    %arg_bias    = OpVariable %_ptr_bias UniformConstant
    %arg_output  = OpVariable %_ptr_output UniformConstant
    %graph_type = OpTypeGraphARM 3 %input_type %input_type %bias_type %output_type
    OpGraphEntryPointARM %graph "MAXPOOL2D_CONV2D" %arg_input %arg_weight %arg_bias %arg_output
    %graph = OpGraphARM %graph_type
    %g_input  = OpGraphInputARM %input_type %uint_0
    %g_weight = OpGraphInputARM %input_type %uint_1
    %g_bias   = OpGraphInputARM %bias_type %uint_2
    %mp_output = OpExtInst %input_type %1 MAX_POOL2D %kernel %stride %pad %nan_mode %g_input
    %conv_output =   OpExtInst %output_type %1 CONV2D %pad %stride %dilation %acc_type %local_bound %mp_output %g_weight %g_bias %input_zero_point %weight_zero_point
    OpGraphSetOutputARM %conv_output %uint_0
    OpGraphEndARM
    ; CHECK: [[input0_shape:%\w+]] = OpConstantComposite %_arr_uint_uint_4 %uint_1 %uint_32 %uint_32 %uint_16
    ; CHECK: [[input0_type:%\w+]] = OpTypeTensorARM %uchar %uint_4 [[input0_shape]]
    ; CHECK: [[_ptr_UniformConstant_input0:%\w+]] = OpTypePointer UniformConstant [[input0_type]]
    ; CHECK: [[var_tensor_in0:%\w+]] = OpVariable [[_ptr_UniformConstant_input0]] UniformConstant
    ; CHECK: [[input1_shape:%\w+]] = OpConstantComposite %_arr_uint_uint_4 %uint_16 %uint_2 %uint_2 %uint_16
    ; CHECK: [[input1_type:%\w+]] = OpTypeTensorARM %uchar %uint_4 [[input1_shape]]
    ; CHECK: [[_ptr_UniformConstant_input1:%\w+]] = OpTypePointer UniformConstant [[input1_type]]
    ; CHECK: [[var_tensor_in1:%\w+]] = OpVariable [[_ptr_UniformConstant_input1]] UniformConstant
    ; CHECK: [[input2_shape:%\w+]] = OpConstantComposite %_arr_uint_uint_1 %uint_16
    ; CHECK: [[input2_type:%\w+]] = OpTypeTensorARM %uint %uint_1 [[input2_shape]]
    ; CHECK: [[_ptr_UniformConstant_input2:%\w+]] = OpTypePointer UniformConstant [[input2_type]]
    ; CHECK: [[var_tensor_in2:%\w+]] = OpVariable [[_ptr_UniformConstant_input2]] UniformConstant
    ; CHECK: [[intermediate_shape:%\w+]] = OpConstantComposite %_arr_uint_uint_4 %uint_1 %uint_16 %uint_16 %uint_16
    ; CHECK: [[intermediate_type:%\w+]] = OpTypeTensorARM %uchar %uint_4 [[intermediate_shape]]
    ; CHECK: [[output0_shape:%\w+]] = OpConstantComposite %_arr_uint_uint_4 %uint_1 %uint_8 %uint_8 %uint_16
    ; CHECK: [[output0_type:%\w+]] = OpTypeTensorARM %uint %uint_4 [[output0_shape]]
    ; CHECK-DAG: [[_ptr_UniformConstant_output0:%\w+]] = OpTypePointer UniformConstant [[output0_type]]
    ; CHECK-DAG: [[var_tensor_out0:%\w+]] = OpVariable [[_ptr_UniformConstant_output0]] UniformConstant
    ; CHECK-DAG: [[graph_type:%\w+]] = OpTypeGraphARM 3 [[input0_type]] [[input1_type]] [[input2_type]] [[output0_type]]
    ; CHECK: OpGraphEntryPointARM [[graph:%\w+]] "MAXPOOL2D_CONV2D" [[var_tensor_in0]] [[var_tensor_in1]] [[var_tensor_in2]] [[var_tensor_out0]]
    ; CHECK: [[graph]] = OpGraphARM [[graph_type]]
    ; CHECK: [[graph_input0:%\w+]] = OpGraphInputARM [[input0_type]] %uint_0
    ; CHECK: [[graph_input1:%\w+]] = OpGraphInputARM [[input1_type]] %uint_1
    ; CHECK: [[graph_input2:%\w+]] = OpGraphInputARM [[input2_type]] %uint_2
    ; CHECK: [[intermediate_output:%\w+]] = OpExtInst [[intermediate_type]] %1 MAX_POOL2D {{.*}} [[graph_input0]]
    ; CHECK: [[graph_output:%\w+]] = OpExtInst [[output0_type]] %1 CONV2D {{.*}} [[intermediate_output]] [[graph_input1]] [[graph_input2]]
    ; CHECK: OpGraphSetOutputARM [[graph_output]] %uint_0
    ; CHECK: OpGraphEndARM

  )";

  const MapOfInterfaceTensorShapes shapes = {
      {{0, 0}, {1, 32, 32, 16}}, // input
      {{0, 1}, {16, 2, 2, 16}}, // weights
      {{0, 2}, {16}}, // bias
  };

  auto result = SinglePassRunAndMatch<GraphShapePass>(
      shader, /* do_validate = */ true, shapes);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

TEST_F(GraphShapeMultiOpTestPass, Conv2d_Conv2d_Concat) {
  std::string shader = header;
  shader += "\n";
  shader += "%1 = OpExtInstImport \"TOSA.001000.1\"";
  shader += R"(
    OpDecorate %CONV2D_0_arg_0 Binding 0
    OpDecorate %CONV2D_0_arg_0 DescriptorSet 0
    OpDecorate %CONV2D_0_arg_1 Binding 1
    OpDecorate %CONV2D_0_arg_1 DescriptorSet 0
    OpDecorate %CONV2D_0_arg_2 Binding 2
    OpDecorate %CONV2D_0_arg_2 DescriptorSet 0
    OpDecorate %CONCAT_res_0 Binding 3
    OpDecorate %CONCAT_res_0 DescriptorSet 0
    OpDecorate %CONV2D_1_arg_0 Binding 0
    OpDecorate %CONV2D_1_arg_0 DescriptorSet 1
    OpDecorate %CONV2D_1_arg_1 Binding 1
    OpDecorate %CONV2D_1_arg_1 DescriptorSet 1
    OpDecorate %CONV2D_1_arg_2 Binding 2
    OpDecorate %CONV2D_1_arg_2 DescriptorSet 1
  )";

  shader += definitions;

  shader += R"(
    %shape1 = OpConstantComposite %_arr_uint_uint_1 %uint_1
    %zero_point_type = OpTypeTensorARM %uchar %uint_1 %shape1
    %axis = OpConstant %uint 3
    %shape2 = OpConstantComposite %_arr_uint_uint_1 %uint_2
    %shape4 = OpConstantComposite %_arr_uint_uint_1 %uint_4
    %tensor_shape2 = OpTypeTensorARM %uint %uint_1 %shape2
    %tensor_shape4 = OpTypeTensorARM %uint %uint_1 %shape4
    %pad = OpConstantComposite %tensor_shape4 %uint_0 %uint_0 %uint_0 %uint_0
    %stride = OpConstantComposite %tensor_shape2 %uint_2 %uint_2
    %dilation = OpConstantComposite %tensor_shape2 %uint_1 %uint_1
    %acc_type = OpConstant %uint 1
    %input_zero_point = OpConstantNull %zero_point_type
    %weight_zero_point = OpConstantNull %zero_point_type
    %local_bound = OpConstantFalse %bool
    %input0_CONV2D_0_type = OpTypeTensorARM %uchar %uint_4
    %_ptr_UniformConstant_input0_CONV2D_0 = OpTypePointer UniformConstant %input0_CONV2D_0_type
    %input1_CONV2D_0_type = OpTypeTensorARM %uint %uint_1
    %_ptr_UniformConstant_input1_CONV2D_0 = OpTypePointer UniformConstant %input1_CONV2D_0_type
    %output0_CONV2D_0_type = OpTypeTensorARM %uint %uint_4
    %CONV2D_0_arg_0 = OpVariable %_ptr_UniformConstant_input0_CONV2D_0 UniformConstant
    %CONV2D_0_arg_1 = OpVariable %_ptr_UniformConstant_input0_CONV2D_0 UniformConstant
    %CONV2D_0_arg_2 = OpVariable %_ptr_UniformConstant_input1_CONV2D_0 UniformConstant
    %_ptr_UniformConstant_input0_CONV2D_1 = OpTypePointer UniformConstant %input0_CONV2D_0_type
    %_ptr_UniformConstant_input1_CONV2D_1 = OpTypePointer UniformConstant %input1_CONV2D_0_type
    %CONV2D_1_arg_0 = OpVariable %_ptr_UniformConstant_input0_CONV2D_1 UniformConstant
    %CONV2D_1_arg_1 = OpVariable %_ptr_UniformConstant_input0_CONV2D_1 UniformConstant
    %CONV2D_1_arg_2 = OpVariable %_ptr_UniformConstant_input1_CONV2D_1 UniformConstant
    %_ptr_UniformConstant_output0_CONCAT = OpTypePointer UniformConstant %output0_CONV2D_0_type
    %CONCAT_res_0 = OpVariable %_ptr_UniformConstant_output0_CONCAT UniformConstant
    %graph_type = OpTypeGraphARM 6 %input0_CONV2D_0_type %input0_CONV2D_0_type %input1_CONV2D_0_type %input0_CONV2D_0_type %input0_CONV2D_0_type %input1_CONV2D_0_type %output0_CONV2D_0_type
    OpGraphEntryPointARM %graph "CONV2D_CONV2D_CONCAT" %CONV2D_0_arg_0 %CONV2D_0_arg_1 %CONV2D_0_arg_2 %CONV2D_1_arg_0 %CONV2D_1_arg_1 %CONV2D_1_arg_2 %CONCAT_res_0
    %graph = OpGraphARM %graph_type
    %graph_input0_CONV2D_0 = OpGraphInputARM %input0_CONV2D_0_type %uint_0
    %graph_input1_CONV2D_0 = OpGraphInputARM %input0_CONV2D_0_type %uint_1
    %graph_input2 = OpGraphInputARM %input1_CONV2D_0_type %uint_2
    %graph_input3_CONV2D_1 = OpGraphInputARM %input0_CONV2D_0_type %uint_3
    %graph_input4_CONV2D_1 = OpGraphInputARM %input0_CONV2D_0_type %uint_4
    %graph_input5 = OpGraphInputARM %input1_CONV2D_0_type %uint_5
    %conv2d_0_output = OpExtInst %output0_CONV2D_0_type %1 CONV2D %pad %stride %dilation %acc_type %local_bound %graph_input0_CONV2D_0 %graph_input1_CONV2D_0 %graph_input2 %input_zero_point %weight_zero_point
    %conv2d_1_output = OpExtInst %output0_CONV2D_0_type %1 CONV2D %pad %stride %dilation %acc_type %local_bound %graph_input3_CONV2D_1 %graph_input4_CONV2D_1 %graph_input5 %input_zero_point %weight_zero_point
    %graph_output = OpExtInst %output0_CONV2D_0_type %1 CONCAT %axis %conv2d_0_output %conv2d_1_output
    OpGraphSetOutputARM %graph_output %uint_0
    OpGraphEndARM
    ; CHECK: [[input0_shape:%\w+]] = OpConstantComposite %_arr_uint_uint_4 %uint_1 %uint_16 %uint_16 %uint_16
    ; CHECK: [[input0_type:%\w+]] = OpTypeTensorARM %uchar %uint_4 [[input0_shape]]
    ; CHECK: [[_ptr_UniformConstant_input0:%\w+]] = OpTypePointer UniformConstant [[input0_type]]
    ; CHECK: [[var_tensor_in0:%\w+]] = OpVariable [[_ptr_UniformConstant_input0]] UniformConstant
    ; CHECK: [[input1_shape:%\w+]] = OpConstantComposite %_arr_uint_uint_4 %uint_16 %uint_2 %uint_2 %uint_16
    ; CHECK: [[input1_type:%\w+]] = OpTypeTensorARM %uchar %uint_4 [[input1_shape]]
    ; CHECK: [[_ptr_UniformConstant_input1:%\w+]] = OpTypePointer UniformConstant [[input1_type]]
    ; CHECK: [[var_tensor_in1:%\w+]] = OpVariable [[_ptr_UniformConstant_input1]] UniformConstant
    ; CHECK: [[input2_shape:%\w+]] = OpConstantComposite %_arr_uint_uint_1 %uint_16
    ; CHECK: [[input2_type:%\w+]] = OpTypeTensorARM %uint %uint_1 [[input2_shape]]
    ; CHECK: [[_ptr_UniformConstant_input2:%\w+]] = OpTypePointer UniformConstant [[input2_type]]
    ; CHECK: [[var_tensor_in2:%\w+]] = OpVariable [[_ptr_UniformConstant_input2]] UniformConstant
    ; CHECK: [[input3_shape:%\w+]] = OpConstantComposite %_arr_uint_uint_4 %uint_1 %uint_16 %uint_16 %uint_72
    ; CHECK: [[input3_type:%\w+]] = OpTypeTensorARM %uchar %uint_4 [[input3_shape]]
    ; CHECK: [[_ptr_UniformConstant_input3:%\w+]] = OpTypePointer UniformConstant [[input3_type]]
    ; CHECK: [[var_tensor_in3:%\w+]] = OpVariable [[_ptr_UniformConstant_input3]] UniformConstant
    ; CHECK: [[input4_shape:%\w+]] = OpConstantComposite %_arr_uint_uint_4 %uint_72 %uint_2 %uint_2 %uint_72
    ; CHECK: [[input4_type:%\w+]] = OpTypeTensorARM %uchar %uint_4 [[input4_shape]]
    ; CHECK: [[_ptr_UniformConstant_input4:%\w+]] = OpTypePointer UniformConstant [[input4_type]]
    ; CHECK: [[var_tensor_in4:%\w+]] = OpVariable [[_ptr_UniformConstant_input4]] UniformConstant
    ; CHECK: [[var_tensor_in5:%\w+]] = OpVariable [[_ptr_UniformConstant_input2]] UniformConstant
    ; CHECK: [[intermediate_shape0:%\w+]] = OpConstantComposite %_arr_uint_uint_4 %uint_1 %uint_8 %uint_8 %uint_16
    ; CHECK: [[intermediate_type0:%\w+]] = OpTypeTensorARM %uint %uint_4 [[intermediate_shape0]]
    ; CHECK: [[intermediate_shape1:%\w+]] = OpConstantComposite %_arr_uint_uint_4 %uint_1 %uint_8 %uint_8 %uint_72
    ; CHECK: [[intermediate_type1:%\w+]] = OpTypeTensorARM %uint %uint_4 [[intermediate_shape1]]
    ; CHECK: [[output0_shape:%\w+]] = OpConstantComposite %_arr_uint_uint_4 %uint_1 %uint_8 %uint_8 %uint_88
    ; CHECK: [[output0_type:%\w+]] = OpTypeTensorARM %uint %uint_4 [[output0_shape]]
    ; CHECK-DAG: [[_ptr_UniformConstant_output0:%\w+]] = OpTypePointer UniformConstant [[output0_type]]
    ; CHECK-DAG: [[var_tensor_out0:%\w+]] = OpVariable [[_ptr_UniformConstant_output0]] UniformConstant
    ; CHECK-DAG: [[graph_type:%\w+]] = OpTypeGraphARM 6 [[input0_type]] [[input1_type]] [[input2_type]] [[input3_type]] [[input4_type]] [[input2_type]] [[output0_type]]
    ; CHECK: OpGraphEntryPointARM [[graph:%\w+]] "CONV2D_CONV2D_CONCAT" [[var_tensor_in0]] [[var_tensor_in1]] [[var_tensor_in2]] [[var_tensor_in3]] [[var_tensor_in4]] [[var_tensor_in5]] [[var_tensor_out0]]
    ; CHECK: [[graph]] = OpGraphARM [[graph_type]]
    ; CHECK: [[graph_input0:%\w+]] = OpGraphInputARM [[input0_type]] %uint_0
    ; CHECK: [[graph_input1:%\w+]] = OpGraphInputARM [[input1_type]] %uint_1
    ; CHECK: [[graph_input2:%\w+]] = OpGraphInputARM [[input2_type]] %uint_2
    ; CHECK: [[graph_input3:%\w+]] = OpGraphInputARM [[input3_type]] %uint_3
    ; CHECK: [[graph_input4:%\w+]] = OpGraphInputARM [[input4_type]] %uint_4
    ; CHECK: [[graph_input5:%\w+]] = OpGraphInputARM [[input2_type]] %uint_5
    ; CHECK: [[intermediate0_output:%\w+]] = OpExtInst [[intermediate_type0]] %1 CONV2D {{.*}} [[graph_input0]] [[graph_input1]] [[graph_input2]]
    ; CHECK: [[intermediate1_output:%\w+]] = OpExtInst [[intermediate_type1]] %1 CONV2D {{.*}} [[graph_input3]] [[graph_input4]] [[graph_input5]]
    ; CHECK: [[graph_output:%\w+]] = OpExtInst [[output0_type]] %1 CONCAT {{.*}} [[intermediate0_output]] [[intermediate1_output]]
    ; CHECK: OpGraphSetOutputARM [[graph_output]] %uint_0
    ; CHECK: OpGraphEndARM

  )";

  const MapOfInterfaceTensorShapes shapes = {
      {{0, 0}, {1, 16, 16, 16}}, // input0
      {{0, 1}, {16, 2, 2, 16}}, // weights0
      {{0, 2}, {16}}, // bias0
      {{1, 0}, {1, 16, 16, 72}}, // input1
      {{1, 1}, {72, 2, 2, 72}}, // weights1
      {{1, 2}, {16}}, // bias1
  };

  auto result = SinglePassRunAndMatch<GraphShapePass>(
      shader, /* do_validate = */ true, shapes);
  EXPECT_EQ(std::get<1>(result), Pass::Status::SuccessWithChange);
}

}  // namespace opt
}  // namespace spvtools