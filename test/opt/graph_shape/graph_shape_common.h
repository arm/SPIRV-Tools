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

#ifndef GRAPH_SHAPE_COMMON_H_
#define GRAPH_SHAPE_COMMON_H_

#include <string>

#include "opt/pass_fixture.h"
#include "source/opt/graph_shape_pass.h"

namespace spvtools {
namespace opt {

const std::string header = R"(
                    OpCapability VulkanMemoryModel
                    OpCapability Shader
                    OpCapability Int8
                    OpCapability Int16
                    OpCapability Int64
                    OpCapability Matrix
                    OpCapability GraphARM
                    OpCapability TensorsARM
                    OpExtension "SPV_KHR_vulkan_memory_model"
                    OpExtension "SPV_ARM_graph"
                    OpExtension "SPV_ARM_tensors"
                    OpMemoryModel Logical Vulkan)";

const std::string definitions = R"(
           %uchar = OpTypeInt 8 0
            %uint = OpTypeInt 32 0
          %uint_0 = OpConstant %uint 0
          %uint_1 = OpConstant %uint 1
          %uint_2 = OpConstant %uint 2
          %uint_3 = OpConstant %uint 3
          %uint_4 = OpConstant %uint 4
          %uint_5 = OpConstant %uint 5
          %uint_8 = OpConstant %uint 8
         %uint_12 = OpConstant %uint 12
         %uint_16 = OpConstant %uint 16
        %uint_256 = OpConstant %uint 256
        %uint_4294967295 = OpConstant %uint 4294967295
%_arr_uint_uint_1 = OpTypeArray %uint %uint_1
%_arr_uint_uint_2 = OpTypeArray %uint %uint_2
%_arr_uint_uint_3 = OpTypeArray %uint %uint_3
%_arr_uint_uint_4 = OpTypeArray %uint %uint_4
            %bool = OpTypeBool
           %float = OpTypeFloat 32)";

struct GraphShapeSingleOpCase {
  std::string OpNameUpperCase;
  std::vector<std::string> insElemTypes;
  std::vector<std::string> outsElemTypes;
  MapOfInterfaceTensorShapes insShapes;
  MapOfInterfaceTensorShapes outsShapes;
  std::vector<std::pair<std::string, std::string>> attributes;
  std::vector<std::pair<std::string, std::string>> postInputAttributes;
  std::string extras;
  std::string tosaVersion;
};

using GraphShapeSingleOpTestPass =
    PassTest<::testing::TestWithParam<GraphShapeSingleOpCase>>;
using GraphShapeSingleOpTestFail =
    PassTest<::testing::TestWithParam<GraphShapeSingleOpCase>>;

}  // namespace opt
}  // namespace spvtools
#endif
