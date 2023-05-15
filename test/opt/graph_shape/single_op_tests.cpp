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

#include <vector>

#include "gtest/gtest-param-test.h"
#include "test/opt/graph_shape/graph_shape_common.h"
namespace spvtools {
namespace opt {
namespace {

const MapOfInterfaceTensorShapes unaryIn = {
    {{0, 0}, {1, 16, 16, 3}},
};
const MapOfInterfaceTensorShapes binaryIns = {
    {{0, 0}, {1, 16, 16, 3}},
    {{0, 1}, {1, 16, 16, 3}},
};
const MapOfInterfaceTensorShapes unaryOutsMatch = {
    {{0, 1}, {1, 16, 16, 3}},
};
const MapOfInterfaceTensorShapes binaryOutsMatch = {
    {{0, 2}, {1, 16, 16, 3}},
};

std::vector<GraphShapeSingleOpCase> passTests = {
    // Tensor Operators
    {
        "ARGMAX",
        {"float"},
        {"uint"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 4}},
        },
        {
            {"axis", "OpConstant %uint 2"},
            {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "AVG_POOL2D",
        {"uchar"},
        {"uchar"},
        {
            {{0, 0}, {1, 16, 16, 3}},
        },
        {
            {{0, 1}, {1, 14, 14, 3}},
        },
        {
            {"kernel", "OpConstantComposite %tensor_shape2 %uint_3 %uint_3"},
            {"stride", "OpConstantComposite %tensor_shape2 %uint_1 %uint_1"},
            {"pad",
             "OpConstantComposite %tensor_shape4 %uint_0 %uint_0 %uint_0 "
             "%uint_0"},
            {"acc_size", "OpConstant %uint 0"},
        },
        {
            {"input_zp", "OpConstantComposite %zero_point_type %uchar_254"},
            {"output_zp", "OpConstantComposite %zero_point_type %uchar_2"},
        },
        R"(%shape2 = OpConstantComposite %_arr_uint_uint_1 %uint_2
           %shape4 = OpConstantComposite %_arr_uint_uint_1 %uint_4
           %tensor_shape2 = OpTypeTensorARM %uint %uint_1 %shape2
           %tensor_shape4 = OpTypeTensorARM %uint %uint_1 %shape4
           %shape1 = OpConstantComposite %_arr_uint_uint_1 %uint_1
           %zero_point_type = OpTypeTensorARM %uchar %uint_1 %shape1
           %uchar_254 = OpConstant %uchar 254
           %uchar_2   = OpConstant %uchar 2)",
        "TOSA.001000.1",
    },
    {
        "MAX_POOL2D",
        {"uchar"},
        {"uchar"},
        {
            {{0, 0}, {1, 16, 16, 16}},
        },
        {
            {{0, 1}, {1, 8, 8, 16}},
        },
        {
            {"kernel", "OpConstantComposite %tensor_shape2 %uint_2 %uint_2"},
            {"stride", "OpConstantComposite %tensor_shape2 %uint_2 %uint_2"},
            {"pad",
             "OpConstantComposite %tensor_shape4 %uint_0 %uint_0 %uint_0 "
             "%uint_0"},
             {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        R"(%shape2 = OpConstantComposite %_arr_uint_uint_1 %uint_2
           %shape4 = OpConstantComposite %_arr_uint_uint_1 %uint_4
           %tensor_shape2 = OpTypeTensorARM %uint %uint_1 %shape2
           %tensor_shape4 = OpTypeTensorARM %uint %uint_1 %shape4)",
        "TOSA.001000.1",
    },
    // Activation Operators
    {
        "CLAMP",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsMatch,
        {
            {"min_val", "OpConstant %float 0"},
            {"max_val", "OpConstant %float 10"},
            {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "ERF",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "SIGMOID",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "TANH",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    // Unary Elementwise Operators
    {
        "ABS",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "BITWISE_NOT",
        {"uint"},
        {"uint"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "CEIL",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "CLZ",
        {"uint"},
        {"uint"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "EXP",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "FLOOR",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOG",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOGICAL_NOT",
        {"bool"},
        {"bool"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "NEGATE",
        {"uint"},
        {"uint"},
        unaryIn,
        unaryOutsMatch,
        {},
        {
            {"input_zp", "OpConstantComposite %zero_point_type %uint_0"},
            {"output_zp", "OpConstantComposite %zero_point_type %uint_0"},
        },
        R"(%shape1 = OpConstantComposite %_arr_uint_uint_1 %uint_1
           %zero_point_type = OpTypeTensorARM %uint %uint_1 %shape1)",
        "TOSA.001000.1",
    },
    {
        "RECIPROCAL",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "RSQRT",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    // Binary Elementwise Operators
    {
        "ADD",
        {"float", "float"},
        {"float"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "ARITHMETIC_RIGHT_SHIFT",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsMatch,
        {
            {"round", "OpConstantTrue %bool"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "BITWISE_AND",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "BITWISE_OR",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "BITWISE_XOR",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "INTDIV",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOGICAL_AND",
        {"bool", "bool"},
        {"bool"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOGICAL_LEFT_SHIFT",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOGICAL_RIGHT_SHIFT",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOGICAL_OR",
        {"bool", "bool"},
        {"bool"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOGICAL_XOR",
        {"bool", "bool"},
        {"bool"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "MAXIMUM",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsMatch,
        {
            {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "MINIMUM",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsMatch,
        {
            {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "MUL",
        {"float", "float"},
        {"float"},
        binaryIns,
        binaryOutsMatch,
        {},
        {
            {"shift", "OpConstantComposite %shift_type %uchar_2"},
        },
        R"(%shape1 = OpConstantComposite %_arr_uint_uint_1 %uint_1
           %shift_type = OpTypeTensorARM %uchar %uint_1 %shape1
           %uchar_2 = OpConstant %uchar 2)",
        "TOSA.001000.1",
    },
    {
        "POW",
        {"float", "float"},
        {"float"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "SUB",
        {"float", "float"},
        {"float"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "TABLE",
        {"uchar"},
        {"uchar"},
        {
            {{0, 0}, {1, 4, 4, 4}},
        },
        {
            {{0, 1}, {1, 4, 4, 4}},
        },
        {},
        {
            {"table", "OpGraphConstantARM %table_type 0"},
        },
        R"(%table_shape = OpConstantComposite %_arr_uint_uint_1 %uint_256
           %table_type = OpTypeTensorARM %uchar %uint_1 %table_shape)",
        "TOSA.001000.1",
    },
    // Elementwise-ternary operators
    {
        "SELECT",
        {"bool", "float", "float"},
        {"float"},
        {
            {{0, 0}, {1, 16, 16, 3}},
            {{0, 1}, {1, 16, 16, 3}},
            {{0, 2}, {1, 16, 16, 3}},
        },
        {
            {{0, 3}, {1, 16, 16, 3}},
        },
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    // Comparison Operators
    {
        "EQUAL",
        {"uint", "uint"},
        {"bool"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "GREATER",
        {"uint", "uint"},
        {"bool"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "GREATER_EQUAL",
        {"uint", "uint"},
        {"bool"},
        binaryIns,
        binaryOutsMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    // Reduction Operators
    {
        "REDUCE_ALL",
        {"bool"},
        {"bool"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 16, 1}},
        },
        {
            {"axis", "OpConstant %uint 3"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "REDUCE_ANY",
        {"bool"},
        {"bool"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 16, 1}},
        },
        {
            {"axis", "OpConstant %uint 3"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "REDUCE_MAX",
        {"uchar"},
        {"uchar"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 16, 1}},
        },
        {
            {"axis", "OpConstant %uint 3"},
            {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "REDUCE_MIN",
        {"uchar"},
        {"uchar"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 16, 1}},
        },
        {
            {"axis", "OpConstant %uint 3"},
            {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "REDUCE_PRODUCT",
        {"float"},
        {"float"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 16, 1}},
        },
        {
            {"axis", "OpConstant %uint 3"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "REDUCE_SUM",
        {"float"},
        {"float"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 16, 1}},
        },
        {
            {"axis", "OpConstant %uint 3"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "CONV2D",
        {"uchar", "uchar", "uint"},
        {"uint"},
        {
            {{0, 0}, {1, 16, 16, 16}}, // input
            {{0, 1}, {16, 2, 2, 16}}, // weights
            {{0, 2}, {16}}, // bias
        },
        {
            {{0, 3}, {1, 8, 8, 16}}, // output
        },
        {
            {"pad",
             "OpConstantComposite %tensor_shape4 %uint_0 %uint_0 %uint_0 "
             "%uint_0"},
            {"stride", "OpConstantComposite %tensor_shape2 %uint_2 %uint_2"},
            {"dilation",
             "OpConstantComposite %tensor_shape2 %uint_1 %uint_1"},
            {"acc_type", "OpConstant %uint 1"},
            {"local_bound", "OpConstantFalse %bool"},
        },
        {
            {"input_zero_point", "OpConstantNull %zero_point_type"},
            {"weight_zero_point", "OpConstantNull %zero_point_type"},
        },
        R"(%shape2 = OpConstantComposite %_arr_uint_uint_1 %uint_2
           %shape4 = OpConstantComposite %_arr_uint_uint_1 %uint_4
           %tensor_shape2 = OpTypeTensorARM %uint %uint_1 %shape2
           %tensor_shape4 = OpTypeTensorARM %uint %uint_1 %shape4
           %shape1 = OpConstantComposite %_arr_uint_uint_1 %uint_1
           %zero_point_type = OpTypeTensorARM %uchar %uint_1 %shape1)",
        "TOSA.001000.1",
    },
    {
       "RESIZE",
       {"uchar"},
       {"uchar"},
       {
           {{0, 0}, {1, 16, 16, 16}}, // input
       },
       {
           {{0, 1}, {1, 32, 32, 16}}, // output
       },
       {
           {"mode", "OpConstant %uint 0"},
       },
       {
           {"scale", "OpConstantComposite %tensor_scale %uint_2 %uint_1 %uint_2 %uint_1"},
           {"offset", "OpConstantComposite %tensor_offset %uint_4294967295 %uint_4294967295"},
           {"border", "OpConstantComposite %tensor_border %uint_0 %uint_0"},
       },
       R"(%scale_vals = OpConstantComposite %_arr_uint_uint_4 %uint_2 %uint_1 %uint_2 %uint_1
        %offset_vals = OpConstantComposite %_arr_uint_uint_2 %uint_4294967295 %uint_4294967295
        %border_vals = OpConstantComposite %_arr_uint_uint_2 %uint_0 %uint_0
        %scale_shape = OpConstantComposite %_arr_uint_uint_1 %uint_4
        %tensor_scale = OpTypeTensorARM %uint %uint_1 %scale_shape
        %offset_shape = OpConstantComposite %_arr_uint_uint_1 %uint_2
        %tensor_offset = OpTypeTensorARM %uint %uint_1 %offset_shape
        %border_shape = OpConstantComposite %_arr_uint_uint_1 %uint_2
        %tensor_border = OpTypeTensorARM %uint %uint_1 %border_shape)",
        "TOSA.001000.1",
    },
    {
        "CONCAT",
        {"uchar", "uchar", "uchar" },
        {"uchar"},
        {
            {{0, 0}, {1, 16, 16, 16}}, // input 0
            {{0, 1}, {1, 16, 16, 3 }}, // input 1
            {{0, 2}, {1, 16, 16, 12}}, // input 2
        },
        {
            {{0, 3}, {1, 16, 16, 31}}, // output
        },
        {
            {"axis", "OpConstant %uint 3"}, // Concat along last dimension (C)
        },
        {},
        "",
        "TOSA.001000.1",
    },
};

INSTANTIATE_TEST_SUITE_P(UnshapedOutputTests, GraphShapeSingleOpTestPass,
                         ::testing::ValuesIn(passTests));

const MapOfInterfaceTensorShapes unaryOutsNotMatch = {
    {{0, 1}, {1, 12, 8, 12}},
};

const MapOfInterfaceTensorShapes binaryOutsNotMatch = {
    {{0, 2}, {1, 12, 8, 12}},
};

std::vector<GraphShapeSingleOpCase> failTests = {
    // Tensor Operators
    {
        "ARGMAX",
        {"float"},
        {"uint"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 16}},
        },
        {
            {"axis", "OpConstant %uint 2"},
            {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "AVG_POOL2D",
        {"uchar"},
        {"uchar"},
        {
            {{0, 0}, {1, 16, 16, 3}},
        },
        {
            {{0, 1}, {1, 12, 12, 2}},
        },
        {
            {"kernel", "OpConstantComposite %tensor_shape2 %uint_3 %uint_3"},
            {"stride", "OpConstantComposite %tensor_shape2 %uint_1 %uint_1"},
            {"pad",
             "OpConstantComposite %tensor_shape4 %uint_0 %uint_0 %uint_0 "
             "%uint_0"},
            {"acc_size", "OpConstant %uint 0"},
        },
        {
            {"input_zp", "OpConstantComposite %zero_point_type %uchar_254"},
            {"output_zp", "OpConstantComposite %zero_point_type %uchar_2"},
        },
        R"(%shape2 = OpConstantComposite %_arr_uint_uint_1 %uint_2
           %shape4 = OpConstantComposite %_arr_uint_uint_1 %uint_4
           %tensor_shape2 = OpTypeTensorARM %uint %uint_1 %shape2
           %tensor_shape4 = OpTypeTensorARM %uint %uint_1 %shape4
           %shape1 = OpConstantComposite %_arr_uint_uint_1 %uint_1
           %zero_point_type = OpTypeTensorARM %uchar %uint_1 %shape1
           %uchar_254 = OpConstant %uchar 254
           %uchar_2   = OpConstant %uchar 2)",
        "TOSA.001000.1",
    },
    {
        "MAX_POOL2D",
        {"uchar"},
        {"uchar"},
        {
            {{0, 0}, {1, 16, 16, 16}},
        },
        {
            {{0, 1}, {1, 16, 8, 16}},
        },
        {
            {"kernel", "OpConstantComposite %tensor_shape2 %uint_2 %uint_2"},
            {"stride", "OpConstantComposite %tensor_shape2 %uint_2 %uint_2"},
            {"pad",
             "OpConstantComposite %tensor_shape4 %uint_0 %uint_0 %uint_0 "
             "%uint_0"},
             {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        R"(%shape2 = OpConstantComposite %_arr_uint_uint_1 %uint_2
           %shape4 = OpConstantComposite %_arr_uint_uint_1 %uint_4
           %tensor_shape2 = OpTypeTensorARM %uint %uint_1 %shape2
           %tensor_shape4 = OpTypeTensorARM %uint %uint_1 %shape4)",
        "TOSA.001000.1",
    },
    // Activation Operators
    {
        "CLAMP",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsNotMatch,
        {
            {"min_val", "OpConstant %float 0"},
            {"max_val", "OpConstant %float 10"},
            {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "ERF",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "SIGMOID",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "TANH",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    // Unary Elementwise Operators
    {
        "ABS",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "BITWISE_NOT",
        {"uint"},
        {"uint"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "CEIL",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "CLZ",
        {"uint"},
        {"uint"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "EXP",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "FLOOR",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOG",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOGICAL_NOT",
        {"bool"},
        {"bool"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "NEGATE",
        {"uint"},
        {"uint"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {
            {"input_zp", "OpConstantComposite %zero_point_type %uint_0"},
            {"output_zp", "OpConstantComposite %zero_point_type %uint_0"},
        },
        R"(%shape1 = OpConstantComposite %_arr_uint_uint_1 %uint_1
           %zero_point_type = OpTypeTensorARM %uint %uint_1 %shape1)",
        "TOSA.001000.1",
    },
    {
        "RECIPROCAL",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "RSQRT",
        {"float"},
        {"float"},
        unaryIn,
        unaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    // Binary Elementwise Operators
    {
        "ADD",
        {"float", "float"},
        {"float"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "ARITHMETIC_RIGHT_SHIFT",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsNotMatch,
        {
            {"round", "OpConstantTrue %bool"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "BITWISE_AND",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "BITWISE_OR",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "BITWISE_XOR",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "INTDIV",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOGICAL_AND",
        {"bool", "bool"},
        {"bool"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOGICAL_LEFT_SHIFT",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOGICAL_RIGHT_SHIFT",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOGICAL_OR",
        {"bool", "bool"},
        {"bool"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "LOGICAL_XOR",
        {"bool", "bool"},
        {"bool"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "MAXIMUM",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsNotMatch,
        {
            {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "MINIMUM",
        {"uint", "uint"},
        {"uint"},
        binaryIns,
        binaryOutsNotMatch,
        {
            {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "MUL",
        {"float", "float"},
        {"float"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {
            {"shift", "OpConstantComposite %shift_type %uchar_2"},
        },
        R"(%shape1 = OpConstantComposite %_arr_uint_uint_1 %uint_1
           %shift_type = OpTypeTensorARM %uchar %uint_1 %shape1
           %uchar_2 = OpConstant %uchar 2)",
        "TOSA.001000.1",
    },
    {
        "POW",
        {"float", "float"},
        {"float"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "SUB",
        {"float", "float"},
        {"float"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "TABLE",
        {"uchar"},
        {"uchar"},
        {
            {{0, 0}, {1, 4, 4, 4}},
        },
        {
            {{0, 1}, {1, 8, 8, 8}},
        },
        {},
        {
            {"table", "OpGraphConstantARM %table_type 0"},
        },
        R"(%table_shape = OpConstantComposite %_arr_uint_uint_1 %uint_256
           %table_type = OpTypeTensorARM %uchar %uint_1 %table_shape)",
        "TOSA.001000.1",
    },
    // Elementwise-ternary operators
    {
        "SELECT",
        {"bool", "float", "float"},
        {"float"},
        {
            {{0, 0}, {1, 16, 16, 3}},
            {{0, 1}, {1, 16, 16, 3}},
            {{0, 2}, {1, 16, 16, 3}},
        },
        {
            {{0, 3}, {1, 12, 8, 12}},
        },
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    // Comparison Operators
    {
        "EQUAL",
        {"uint", "uint"},
        {"bool"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "GREATER",
        {"uint", "uint"},
        {"bool"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "GREATER_EQUAL",
        {"uint", "uint"},
        {"bool"},
        binaryIns,
        binaryOutsNotMatch,
        {},
        {},
        "",
        "TOSA.001000.1",
    },
    // Reduction Operators
    {
        "REDUCE_ALL",
        {"bool"},
        {"bool"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 1, 4}},
        },
        {
            {"axis", "OpConstant %uint 3"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "REDUCE_ANY",
        {"bool"},
        {"bool"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 1, 4}},
        },
        {
            {"axis", "OpConstant %uint 3"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "REDUCE_MAX",
        {"uchar"},
        {"uchar"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 1, 4}},
        },
        {
            {"axis", "OpConstant %uint 3"},
            {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "REDUCE_MIN",
        {"uchar"},
        {"uchar"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 1, 4}},
        },
        {
            {"axis", "OpConstant %uint 3"},
            {"nan_mode", "OpConstant %uint 1"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "REDUCE_PRODUCT",
        {"float"},
        {"float"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 1, 4}},
        },
        {
            {"axis", "OpConstant %uint 3"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "REDUCE_SUM",
        {"float"},
        {"float"},
        {
            {{0, 0}, {1, 8, 16, 4}},
        },
        {
            {{0, 1}, {1, 8, 1, 4}},
        },
        {
            {"axis", "OpConstant %uint 3"},
        },
        {},
        "",
        "TOSA.001000.1",
    },
    {
        "CONV2D",
        {"uchar", "uchar", "uint"},
        {"uint"},
        {
            {{0, 0}, {1, 16, 16, 16}}, // input
            {{0, 1}, {16, 2, 2, 16}}, // weights
            {{0, 2}, {16}}, // bias
        },
        {
            {{0, 3}, {1, 8, 4, 16}}, // output
        },
        {
            {"pad",
             "OpConstantComposite %tensor_shape4 %uint_0 %uint_0 %uint_0 "
             "%uint_0"},
            {"stride", "OpConstantComposite %tensor_shape2 %uint_2 %uint_2"},
            {"dilation",
             "OpConstantComposite %tensor_shape2 %uint_1 %uint_1"},
            {"acc_type", "OpConstant %uint 1"},
            {"local_bound", "OpConstantFalse %bool"},
        },
        {
            {"input_zero_point", "OpConstantNull %zero_point_type"},
            {"weight_zero_point", "OpConstantNull %zero_point_type"},
        },
        R"(%shape2 = OpConstantComposite %_arr_uint_uint_1 %uint_2
           %shape4 = OpConstantComposite %_arr_uint_uint_1 %uint_4
           %tensor_shape2 = OpTypeTensorARM %uint %uint_1 %shape2
           %tensor_shape4 = OpTypeTensorARM %uint %uint_1 %shape4
           %shape1 = OpConstantComposite %_arr_uint_uint_1 %uint_1
           %zero_point_type = OpTypeTensorARM %uchar %uint_1 %shape1)",
        "TOSA.001000.1",
    },
    {
       "RESIZE",
       {"uchar"},
       {"uchar"},
       {
           {{0, 0}, {1, 16, 16, 16}}, // input
       },
       {
           {{0, 1}, {1, 16, 16, 16}}, // output
       },
       {
           {"mode", "OpConstant %uint 0"},
       },
       {
           {"scale", "OpConstantComposite %tensor_scale %uint_2 %uint_1 %uint_2 %uint_1"},
           {"offset", "OpConstantComposite %tensor_offset %uint_4294967295 %uint_4294967295"},
           {"border", "OpConstantComposite %tensor_border %uint_0 %uint_0"},
       },
       R"(%scale_vals = OpConstantComposite %_arr_uint_uint_4 %uint_2 %uint_1 %uint_2 %uint_1
        %offset_vals = OpConstantComposite %_arr_uint_uint_2 %uint_4294967295 %uint_4294967295
        %border_vals = OpConstantComposite %_arr_uint_uint_2 %uint_0 %uint_0
        %scale_shape = OpConstantComposite %_arr_uint_uint_1 %uint_4
        %tensor_scale = OpTypeTensorARM %uint %uint_1 %scale_shape
        %offset_shape = OpConstantComposite %_arr_uint_uint_1 %uint_2
        %tensor_offset = OpTypeTensorARM %uint %uint_1 %offset_shape
        %border_shape = OpConstantComposite %_arr_uint_uint_1 %uint_2
        %tensor_border = OpTypeTensorARM %uint %uint_1 %border_shape)",
        "TOSA.001000.1",
    },
    {
        "CONCAT",
        {"uchar", "uchar", "uchar" },
        {"uchar"},
        {
            {{0, 0}, {1, 16, 16, 16}}, // input 0
            {{0, 1}, {1, 16, 16, 3 }}, // input 1
            {{0, 2}, {1, 16, 16, 12}}, // input 2
        },
        {
            {{0, 3}, {1, 16, 16, 16}}, // output
        },
        {
            {"axis", "OpConstant %uint 3"}, // Concat along last dimension (C)
        },
        {},
        "",
        "TOSA.001000.1",
    },
};

INSTANTIATE_TEST_SUITE_P(ExistingOutputShapeNotMatchingInferedOutputShapeTests,
                         GraphShapeSingleOpTestFail,
                         ::testing::ValuesIn(failTests));

}  // namespace
}  // namespace opt
}  // namespace spvtools