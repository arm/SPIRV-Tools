// Copyright (c) 2023-2025 Arm Ltd.
// Copyright (c) 2021 Google LLC
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

#include "source/util/parse_number.h"

namespace spvtools {
namespace opt {

namespace {

// Returns true if the given char is ':', ',', '\0' or considered as blank space
// (i.e.: '\n', '\r', '\v', '\t', '\f' and ' ').
bool IsSeparator(char ch) {
  return std::strchr(":,\0", ch) || std::isspace(ch) != 0;
}

// Reads characters starting from |str| until it meets a separator. Parses a
// number from the characters and stores it into |number|. Returns the pointer
// to the separator if it succeeds. Otherwise, returns nullptr.
template <typename T>
const char* ParseNumberUntilSeparator(const char* str, T* number) {
  const char* number_begin = str;
  while (!IsSeparator(*str)) str++;
  const char* number_end = str;
  std::string number_in_str(number_begin, number_end - number_begin);
  if (!utils::ParseNumber(number_in_str.c_str(), number)) {
    // The descriptor set is not a valid T number.
    return nullptr;
  }
  return str;
}

}  // namespace

void GraphShapePass::LogError(const std::string& msg) const {
  consumer()(SPV_MSG_ERROR, "", {0, 0, 0}, msg.c_str());
}

std::unique_ptr<MapOfInterfaceTensorShapes>
GraphShapePass::ParseInterfaceTensorShapesString(const char* str) {
  if (!str) return nullptr;
  printf("Got options: '%s'\n", str);
  auto interface_tensor_shapes = MakeUnique<MapOfInterfaceTensorShapes>();

  while (std::isspace(*str)) str++;  // skip leading spaces.

  // The parsing loop, break when points to the end.
  while (*str) {
    printf("Parsing descriptor set, '%c'\n", *str);
    // Parse the descriptor set.
    uint32_t descriptor_set = 0;
    str = ParseNumberUntilSeparator(str, &descriptor_set);
    if (str == nullptr) return nullptr;

    // Find the ':', spaces between the descriptor set and the ':' are not
    // allowed.
    if (*str++ != ':') {
      // ':' not found
      return nullptr;
    }

    printf("Parsing binding\n");
    // Parse the binding.
    uint32_t binding = 0;
    str = ParseNumberUntilSeparator(str, &binding);
    if (str == nullptr) return nullptr;

    // Find the ':', spaces between the descriptor set and the ':' are not
    // allowed.
    if (*str++ != ':') {
      // ':' not found
      return nullptr;
    }

    printf("Parsing shape, '%c'\n", *str);
    // We now expect a shape
    if (!std::isdigit(*str)) {
      return nullptr;
    }

    bool parsing_shape = true;
    std::vector<uint64_t> shape;
    while (parsing_shape) {
      printf("Parsing shape dim\n");
      uint64_t dim = 0;
      str = ParseNumberUntilSeparator(str, &dim);
      if (str == nullptr) return nullptr;
      printf("Parsed shape dim %" PRIu64 "\n", dim);
      shape.push_back(dim);

      char next = *str;
      if (next == ',') {
        str++;
        continue;
      } else if (std::isspace(next) || next == '\0') {
        parsing_shape = false;
      } else {
        return nullptr;
      }
    }

    // TODO fail if we already have a shape for that resource
    // TODO fail if the resource is not a tensor or no such resource
    interface_tensor_shapes->insert({{descriptor_set, binding}, shape});

    printf(
        "Parsed interface tensor: descriptor_set = %u, binding = %u, shape = ",
        descriptor_set, binding);
    const char* sep = "";
    for (auto dim : shape) {
      printf("%s%" PRIu64, sep, dim);
      sep = ",";
    }
    printf("\n");
    // Skip trailing spaces.
    while (std::isspace(*str)) str++;
  }

  return interface_tensor_shapes;
}

std::optional<std::pair<uint32_t, uint32_t>> GraphShapePass::GetDSBForVar(
    uint32_t var_id) const {
  // TODO use limits instead of hard-coded numbers
  auto decos =
      context()->get_decoration_mgr()->GetDecorationsFor(var_id, false);
  uint32_t set = 0xFFFFFFFF, binding = 0xFFFFFFFF;
  for (const auto& deco : decos) {
    spv::Decoration d = spv::Decoration(deco->GetSingleWordInOperand(1u));
    if (d == spv::Decoration::DescriptorSet) {
      set = deco->GetSingleWordInOperand(2u);
    } else if (d == spv::Decoration::Binding) {
      binding = deco->GetSingleWordInOperand(2u);
    }
  }
  if (set == 0xFFFFFFFF || binding == 0xFFFFFFFF) {
    return std::nullopt;
  }
  return std::make_pair(set, binding);
}

const analysis::TensorARM* GraphShapePass::GetTensorVariableType(
    uint32_t var_id) const {
  // TODO assert storage class?
  auto defuse = context()->get_def_use_mgr();
  auto var_inst = defuse->GetDef(var_id);
  assert(var_inst);
  auto ptrty = context()->get_type_mgr()->GetType(var_inst->type_id());
  assert(ptrty);
  assert(ptrty->kind() == analysis::Type::Kind::kPointer);
  auto pointee_type = ptrty->AsPointer()->pointee_type();
  assert(pointee_type->kind() == analysis::Type::Kind::kTensorARM);
  return pointee_type->AsTensorARM();
}

std::optional<std::vector<uint64_t>> GraphShapePass::GetTensorVariableShape(
    uint32_t var_id) const {
  auto tensor_type = GetTensorVariableType(var_id);
  return GetTensorTypeShape(context(), tensor_type);
}

static bool tensor_shapes_equal(const std::vector<uint64_t>& lhs,
                                const std::vector<uint64_t>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); i++) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

Pass::Status GraphShapePass::ShapeGraphInstruction(Instruction* inst) {
  switch (inst->opcode()) {
    case spv::Op::OpExtInst: {
      uint32_t iset_id = inst->GetSingleWordInOperand(0);
      auto iset = context()->get_def_use_mgr()->GetDef(iset_id);
      assert(iset->opcode() == spv::Op::OpExtInstImport);
      auto iset_name = iset->GetInOperand(0).AsString();
      if (iset_name == "TOSA.001000.1") {
        return ShapeGraphInstructionTOSA(context(), inst);
      } else {
        assert(false && "Unhandled graph instruction set");
        return Status::Failure;
      }
      break;
    }
    default:
      assert(false && "Unhandled graph instruction");
      return Status::Failure;
  }
}

Pass::Status GraphShapePass::ShapeInterfaceTensorVariable(
    uint32_t var_id, const std::vector<uint64_t>& target_shape) {
  auto existing_shape = GetTensorVariableShape(var_id);

  // Is the tensor already shaped?
  if (existing_shape != std::nullopt) {
    if (tensor_shapes_equal(target_shape, *existing_shape)) {
      // Yes, and the shape is the one provided by the user, nothing to do
      return Status::SuccessWithoutChange;
    } else {
      // Yes, but the shape is not the one provided by the user, fail
      LogError(
          "Tensor already shaped with a shape different from that requested");  // TODO give specifics
      return Status::Failure;
    }
  }

  // The tensor is not already shaped, we need to shape it

  // Get tensor type
  auto existing_tensor_type = GetTensorVariableType(var_id);
  analysis::Type* tensor_type = GetShapedTensorType(
      context(), existing_tensor_type->element_type(), target_shape);

  // Get pointer type
  analysis::Pointer ptr_type(
      tensor_type, spv::StorageClass::UniformConstant);  // FIXME get storage
                                                         // class from variable
  analysis::Type* tensor_pointer_type =
      context()->get_type_mgr()->GetRegisteredType(&ptr_type);

  // Redefine variable
  uint32_t pointer_type_id =
      context()->get_type_mgr()->GetId(tensor_pointer_type);
  uint32_t new_var_id = TakeNextId();
  std::unique_ptr<Instruction> new_var(new Instruction(
      context(), spv::Op::OpVariable, pointer_type_id, new_var_id,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER,
        {uint32_t(spv::StorageClass::UniformConstant)}}}));
  context()->AddGlobalValue(std::move(new_var));
  context()->ReplaceAllUsesWith(var_id, new_var_id);
  context()->KillDef(var_id);

  return Status::SuccessWithChange;
}

Pass::Status GraphShapePass::ProcessEntryPoint(Instruction& epinst) {
  bool changed = false;
  // Get graph
  auto graph_inst =
      context()->get_def_use_mgr()->GetDef(epinst.GetSingleWordInOperand(0));
  auto graph_type =
      context()->get_type_mgr()->GetType(graph_inst->type_id())->AsGraphARM();

  // Shape interface tensors connected to graph inputs
  for (uint32_t i = 2; i < 2 + graph_type->num_inputs(); i++) {
    auto iovar_id = epinst.GetSingleWordInOperand(i);
    auto iovar_ty = GetTensorVariableType(iovar_id);
    assert(iovar_ty);
    auto dsb = GetDSBForVar(iovar_id);
    if (interface_tensor_shapes_.count(*dsb) == 0) {
      LogError(
          "No user-supplied shape for interface tensor");  // TODO specifics
      return Status::Failure;
    }
    auto const& target_shape = interface_tensor_shapes_.at(*dsb);
    // Is the interface tensor already shaped?
    if (iovar_ty->is_shaped()) {
      // Does shape match user request?
      if (tensor_shapes_equal(*GetTensorVariableShape(iovar_id),
                              target_shape)) {
        // Yes, nothing to do for this tensor
        continue;
      } else {
        LogError(
            "Existing interface tensor shape mismatched user supplied shape");  // TODO give specifics
        return Status::Failure;
      }
    }
    // The interface tensor needs shaping
    auto status = ShapeInterfaceTensorVariable(iovar_id, target_shape);
    if (status == Status::Failure) {
      return status;
    }
    if (status == Status::SuccessWithChange) {
      changed = true;
    }
  }

  // Is the graph already shaped?
  if (graph_type->is_shaped()) {
    // If yes, does its shape match the interface tensors'
    for (uint32_t i = 0; i < graph_type->io_types().size(); i++) {
      // No, failure
      auto tensor_type_graph = graph_type->io_types()[i]->AsTensorARM();
      auto tensor_interface_var_id = epinst.GetSingleWordInOperand(2 + i);
      auto tensor_type_interface =
          GetTensorVariableType(tensor_interface_var_id);
      if (!tensor_shapes_equal(
              *GetTensorTypeShape(context(), tensor_type_graph),
              *GetTensorTypeShape(context(), tensor_type_interface))) {
        LogError(
            "Mismatch between existing graph shape and interface tensor shape");  // TODO give specifics
        return Status::Failure;
      }
    }
    // Graph shape matches that of interface tensors, nothing to do
    return Status::SuccessWithoutChange;
  }
  // The graph needs shaping
  // FIXME Is the graph used by another graph and/or entry point
  // Yes, duplicate when necessary
  // No, mutate

  changed = true;
  std::vector<const analysis::Type*> shaped_graph_io_types;

  // Mutate type of graph inputs
  auto graph = context()->GetGraph(graph_inst);
  for (uint32_t i = 0; i < graph->inputs().size(); i++) {
    auto& input = graph->inputs()[i];
    auto iovar_ty = GetTensorVariableType(epinst.GetSingleWordInOperand(i + 2));
    shaped_graph_io_types.push_back(iovar_ty);
    input->SetResultType(
        context()->get_type_mgr()->GetTypeInstruction(iovar_ty));
  }
  // Mutate body instructions
  for (auto& binst : graph->instructions()) {
    auto status = ShapeGraphInstruction(binst.get());
    if (status == Status::Failure) {
      return status;
    }
    if (status == Status::SuccessWithChange) {
      changed = true;
    }
  }
  // Collect graph output types
  auto const& output_insts = graph->outputs();
  for (uint32_t i = 0; i < output_insts.size(); i++) {
    auto const& output_inst = output_insts[i];
    auto output_val_id = output_inst->GetSingleWordInOperand(0);
    auto output_val = context()->get_def_use_mgr()->GetDef(output_val_id);
    auto tensor_type = context()
                           ->get_type_mgr()
                           ->GetType(output_val->type_id())
                           ->AsTensorARM();
    shaped_graph_io_types.push_back(tensor_type);
  }
  // Mutate graph type
  analysis::GraphARM new_gtype(graph_type->num_inputs(), shaped_graph_io_types);
  analysis::Type* new_graph_type =
      context()->get_type_mgr()->GetRegisteredType(&new_gtype);
  graph_inst->SetResultType(
      context()->get_type_mgr()->GetTypeInstruction(new_graph_type));

  // Shape output interface tensors
  for (uint32_t i = 0; i < output_insts.size(); i++) {
    uint32_t output_var_id =
        epinst.GetSingleWordInOperand(2 + graph_type->num_inputs() + i);
    auto target_shape =
        GetTensorShape(context(), output_insts[i]->GetSingleWordInOperand(0));
    auto status = ShapeInterfaceTensorVariable(output_var_id, *target_shape);
    // TODO check that output shape matches user-provided shape for output, if
    // any
    if (status == Status::Failure) {
      return status;
    }
    if (status == Status::SuccessWithChange) {
      changed = true;
    }
  }

  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Pass::Status GraphShapePass::Process() {
  bool changed = false;

  // For all graph entry points
  for (auto& ep : context()->module()->graph_entry_points()) {
    auto status = ProcessEntryPoint(ep);
    if (status == Status::Failure) {
      return status;
    }
    if (status == Status::SuccessWithChange) {
      changed = true;
    }
  }

  return changed ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

std::optional<std::vector<uint64_t>> GetTensorTypeShape(
    IRContext* context, const analysis::TensorARM* tensor_type) {
  if (tensor_type->shape_id() == 0) {
    return std::nullopt;
  }
  auto shapecst = context->get_constant_mgr()
                      ->FindDeclaredConstant(tensor_type->shape_id())
                      ->AsCompositeConstant();
  assert(shapecst);
  std::vector<uint64_t> shape;
  for (auto const c : shapecst->GetComponents()) {
    auto dim = c->GetZeroExtendedValue();
    assert(dim);
    shape.push_back(dim);
  }
  return shape;
}

std::optional<std::vector<uint64_t>> GetTensorShape(IRContext* context,
                                                    uint32_t id) {
  auto inst = context->get_def_use_mgr()->GetDef(id);
  auto tensor_type =
      context->get_type_mgr()->GetType(inst->type_id())->AsTensorARM();
  return GetTensorTypeShape(context, tensor_type);
}

analysis::TensorARM* GetShapedTensorType(
    IRContext* context, const analysis::Type* tensor_elem_type,
    const std::vector<uint64_t>& target_shape) {
  // FIXME check rank
  auto cstmgr = context->get_constant_mgr();
  uint32_t tensor_rank = static_cast<uint32_t>(target_shape.size());
  auto rankcst_id = cstmgr->GetUIntConstId(tensor_rank);
  std::vector<uint32_t> shape_ids;
  for (auto dim : target_shape) {
    // FIXME automatically select integer type and/or assert that values fit in
    // 32-bit
    auto c = cstmgr->GetUIntConstId(static_cast<uint32_t>(dim));
    shape_ids.push_back(c);
  }
  auto tymgr = context->get_type_mgr();
  analysis::Integer i(32, false);
  analysis::Type* shape_array_elem_type =
      context->get_type_mgr()->GetRegisteredType(&i);
  analysis::Array array_type(
      shape_array_elem_type,
      analysis::Array::LengthInfo{rankcst_id, {0, tensor_rank}});

  analysis::Type* shape_arr_type = tymgr->GetRegisteredType(&array_type);

  auto shapecst = cstmgr->GetConstant(shape_arr_type, shape_ids);
  auto shapecst_id = cstmgr->GetDefiningInstruction(shapecst)->result_id();
  analysis::TensorARM tensor_type_tpl(tensor_elem_type, rankcst_id,
                                      shapecst_id);
  return tymgr->GetRegisteredType(&tensor_type_tpl)->AsTensorARM();
}

std::vector<int32_t> GetConstantValuesOfKnownSizeI32(IRContext* context,
                                                     uint32_t id,
                                                     size_t numElements) {
  auto constant = context->get_constant_mgr()->FindDeclaredConstant(id);
  if (constant->AsNullConstant() != nullptr) {
    return std::vector<int32_t>(numElements, 0);
  } else if (const analysis::TensorConstant* tensorConstant =
                 constant->AsTensorConstant()) {
    const std::vector<const analysis::Constant*>& components =
        tensorConstant->GetComponents();
    if (components.size() != numElements) {
      assert(false && "Unexpected number of elements in TensorConstant");
      return {};
    }
    std::vector<int32_t> result;
    result.reserve(numElements);
    for (auto&& component : components) {
      result.push_back(component->GetS32());
    }
    return result;
  } else {
    assert(false && "Unexpected constant type");
    return {};
  }
}

}  // namespace opt
}  // namespace spvtools
