// Copyright (c) 2022-2024 Arm Ltd.
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

#ifndef SOURCE_OPT_GRAPH_H_
#define SOURCE_OPT_GRAPH_H_

namespace spvtools {
namespace opt {

// TODO separate inline function to match other objects
// TODO separate cpp file for implementation
// TODO ForEachInst implemented using WhileEachInst?

struct Graph {
  // Creates a graph instance declared by the given OpGraph instruction
  // |def_inst|.
  explicit Graph(std::unique_ptr<Instruction> def_inst) : def_inst_(std::move(def_inst)) {}

  // The OpGraph instruction that begins the definition of this graph.
  Instruction& DefInst() { return *def_inst_; }
  const Instruction& DefInst() const { return *def_inst_; }

  // Appends an input to this graph.
  void AddInput(std::unique_ptr<Instruction> inst) {
    inputs_.emplace_back(std::move(inst));
  }

  // Appends an instruction to this graph.
  void AddInstruction(std::unique_ptr<Instruction> inst) {
    insts_.emplace_back(std::move(inst));
  }

  // Appends an output to this graph.
  void AddOutput(std::unique_ptr<Instruction> inst) {
    outputs_.emplace_back(std::move(inst));
  }

  // Saves the given graph end instruction.
  void SetGraphEnd(std::unique_ptr<Instruction> end_inst) {
    end_inst_ = std::move(end_inst);
  }

  const std::vector<std::unique_ptr<Instruction>>& instructions() const {
    return insts_;
  }
  
  const std::vector<std::unique_ptr<Instruction>>& inputs() const {
    return inputs_;
  }

  const std::vector<std::unique_ptr<Instruction>>& outputs() const {
    return outputs_;
  }

  // Runs the given function |f| on instructions in this graph, in order,
  // and optionally on debug line instructions that might precede them and
  // non-semantic instructions that succceed the function.
  void ForEachInst(const std::function<void(Instruction*)>& f,
                   bool run_on_debug_line_insts = false,
                   bool run_on_non_semantic_insts = false) {
    (void)run_on_debug_line_insts;
    (void)run_on_non_semantic_insts;

    f(def_inst_.get());

    for (auto & inst : inputs_) {
      f(inst.get());
    }

    for (auto & inst : insts_) {
      f(inst.get());
    }

    for (auto & inst : outputs_) {
      f(inst.get());
    }

    f(end_inst_.get());
  }
  void ForEachInst(const std::function<void(const Instruction*)>& f,
                   bool run_on_debug_line_insts = false,
                   bool run_on_non_semantic_insts = false) const {
    (void)run_on_debug_line_insts;
    (void)run_on_non_semantic_insts;

    f(def_inst_.get());

    for (auto& inst : inputs_) {
      f(inst.get());
    }

    for (auto& inst : insts_) {
      f(inst.get());
    }

    for (auto & inst : outputs_) {
      f(inst.get());
    }

    f(end_inst_.get());
  }

 private:
  // The OpGraph instruction that begins the definition of this graph.
  std::unique_ptr<Instruction> def_inst_;
  // All inputs to this graph.
  std::vector<std::unique_ptr<Instruction>> inputs_;
  // All instructions describing this graph
  std::vector<std::unique_ptr<Instruction>> insts_;
  // All outputs of this graph.
  std::vector<std::unique_ptr<Instruction>> outputs_;
  // The OpGraphEnd instruction.
  std::unique_ptr<Instruction> end_inst_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_GRAPH_H_
