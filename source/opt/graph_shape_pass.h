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

#ifndef SOURCE_OPT_GRAPH_SHAPE_PASS_H_
#define SOURCE_OPT_GRAPH_SHAPE_PASS_H_

#include <optional>

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

using MapOfInterfaceTensorShapes =
    std::map<std::pair<uint32_t, uint32_t>, std::vector<int64_t>>;

class GraphShapePass : public Pass {
 public:
  explicit GraphShapePass(
      const MapOfInterfaceTensorShapes& interface_tensor_shapes)
      : interface_tensor_shapes_(interface_tensor_shapes.begin(),
                                 interface_tensor_shapes.end()) {}

  Status Process() override;
  const char* name() const override { return "graph-shape"; }

  static std::unique_ptr<MapOfInterfaceTensorShapes>
  ParseInterfaceTensorShapesString(const char* str);

 private:
  void LogError(const std::string& msg) const;
  std::optional<std::pair<uint32_t, uint32_t>> GetDSBForVar(
      uint32_t var_id) const;
  const analysis::TensorARM* GetTensorVariableType(uint32_t var_id) const;
  std::optional<std::vector<int64_t>> GetTensorVariableShape(
      uint32_t var_id) const;
  Pass::Status ShapeInterfaceTensorVariable(
      uint32_t var_id, const std::vector<int64_t>& target_shape);
  Pass::Status ShapeGraphInstruction(Instruction* inst);
  Pass::Status ProcessEntryPoint(Instruction& epinst);

  MapOfInterfaceTensorShapes interface_tensor_shapes_;
};
std::optional<std::vector<int64_t>> GetTensorTypeShape(
    IRContext* context, const analysis::TensorARM* tensor_type);
std::optional<std::vector<int64_t>> GetTensorShape(IRContext* context,
                                                    uint32_t id);
analysis::TensorARM* GetShapedTensorType(
    IRContext* context, const analysis::Type* tensor_elem_type,
    const std::vector<int64_t>& target_shape);
std::vector<int32_t> GetConstantValuesOfKnownSizeI32(IRContext* context,
                                                     uint32_t id,
                                                     size_t numElements);
Pass::Status ShapeGraphInstructionTOSA(IRContext* context,
                                                Instruction* inst);

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_GRAPH_SHAPE_PASS_H_
