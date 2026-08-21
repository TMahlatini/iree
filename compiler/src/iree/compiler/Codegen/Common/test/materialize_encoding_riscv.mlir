// Tests for RISC-V targets with static VLEN-based tiles and scalable vectorization.
//
// STATIC mode (default): Expected to use VLEN-based tile sizes computed from zvl* features.
// SCALABLE mode (opt-in): Expected to use smaller base tile sizes that scale with vscale at runtime.
//
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s --check-prefixes=CHECK,STATIC
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SCALABLE

// RISC-V f32 matmul encoding tests.

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_res = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>

// RISC-V64 without V extension - no data tiling (fallback)
func.func @negative_matmul_lowering_f32f32f32_riscv64_no_v_ext(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>, %m : index, %n : index, %k : index) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %lhs encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  %1 = iree_encoding.set_encoding %rhs encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_rhs>
  %2 = iree_encoding.set_encoding %acc encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_res>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%2 : tensor<?x?xf32, #encoding_res>)
      -> tensor<?x?xf32, #encoding_res>
  %4 = iree_encoding.unset_encoding %3 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #encoding_res> -> tensor<?x?xf32>{%m, %n}
  return %4 : tensor<?x?xf32>
}

// RISC-V64 without V extension does not implement data-tiling.
// CHECK-LABEL: func @negative_matmul_lowering_f32f32f32_riscv64_no_v_ext
//   CHECK-NOT:   linalg.pack
//   CHECK-NOT:   linalg.mmt4d
//       CHECK:   %[[RES:.+]] = linalg.matmul
//   CHECK-NOT:   linalg.unpack
//       CHECK:   return %[[RES]]

// RISC-V32 without V extension - no data tiling
func.func @negative_matmul_lowering_f32f32f32_riscv32_no_v_ext(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv32-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %acc, %c0 : tensor<?x?xf32>
  %N = tensor.dim %acc, %c1 : tensor<?x?xf32>
  %K = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %lhs encoding_dims{%M, %N, %K} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  %1 = iree_encoding.set_encoding %rhs encoding_dims{%M, %N, %K} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_rhs>
  %2 = iree_encoding.set_encoding %acc encoding_dims{%M, %N, %K} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_res>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%2 : tensor<?x?xf32, #encoding_res>)
      -> tensor<?x?xf32, #encoding_res>
  %4 = iree_encoding.unset_encoding %3 encoding_dims{%M, %N, %K} : tensor<?x?xf32, #encoding_res> -> tensor<?x?xf32>{%M, %N}
  return %4 : tensor<?x?xf32>
}
// RISC-V32 without V extension does not implement data-tiling.
// CHECK-LABEL: func @negative_matmul_lowering_f32f32f32_riscv32_no_v_ext
//   CHECK-NOT:   linalg.pack
//   CHECK-NOT:   linalg.mmt4d
//       CHECK:   %[[RES:.+]] = linalg.matmul
//   CHECK-NOT:   linalg.unpack
//       CHECK:   return %[[RES]]

// RISC-V64 with V extension - set_encoding for LHS (f32)
// The LHS operand corresponds to the M dimension, which is NOT scalable.
// Therefore, STATIC and SCALABLE produce the same result for LHS.
func.func @matmul_set_encoding_LHS_f32_riscv64(%arg0: tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32, #encoding_lhs> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  return %0 : tensor<?x?xf32, #encoding_lhs>
}
// CHECK-LABEL: func.func @matmul_set_encoding_LHS_f32_riscv64
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK:         %[[PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [7, 1]
// CHECK-SAME:      -> tensor<?x?x7x1xf32>
// CHECK:         return %[[PACK]]

// RISC-V64 with V extension - set_encoding for RHS (f32)
// The RHS operand corresponds to the N dimension, which IS scalable in when scalable vectorization is enabled.
// In static mode: N tile is fixed based on VLEN (16 for VLEN=128)
// In scalable mode: N tile is scalable (8 * vscale)
func.func @matmul_set_encoding_RHS_f32_riscv64(%arg0: tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32, #encoding_rhs> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_rhs>
  return %0 : tensor<?x?xf32, #encoding_rhs>
}
// CHECK-LABEL: func.func @matmul_set_encoding_RHS_f32_riscv64
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-DAG:     %[[PAD:.+]] = arith.constant 0.000000e+00 : f32

// SCALABLE-DAG:  %[[C8:.*]] = arith.constant 8 : index
// SCALABLE-DAG:  %[[VSCALE:.*]] = vector.vscale
// SCALABLE:      %[[C8_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C8]] : index

/// With dynamic dimensions the N dimension is padded up to a multiple of the
/// (possibly scalable) inner tile, so both modes emit a padding value.
// CHECK:         %[[PACK:.+]] = linalg.pack %[[SRC]]
// CHECK-SAME:      padding_value(%[[PAD]] : f32)
// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]

// STATIC-SAME:     inner_tiles = [16, 1]
// STATIC-SAME:     -> tensor<?x?x16x1xf32>

// SCALABLE-SAME:   inner_tiles = [%[[C8_VSCALE]], 1]
// SCALABLE-SAME:   -> tensor<?x?x?x1xf32>

// CHECK:         return %[[PACK]]

// RISC-V64 with V extension - unset_encoding for RESULT (f32)
// The RESULT operand has both M and N dimensions. M is fixed, N differs by mode.
func.func @matmul_unset_encoding_RESULT_f32_riscv64(%arg0: tensor<?x?xf32, #encoding_res>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.unset_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #encoding_res> -> tensor<?x?xf32>{%m, %n}
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @matmul_unset_encoding_RESULT_f32_riscv64
// STATIC-SAME:    %[[INPUT:[a-zA-Z0-9]+]]: tensor<?x?x7x16xf32>
// SCALABLE-SAME:  %[[INPUT:[a-zA-Z0-9]+]]: tensor<?x?x7x?xf32>

// CHECK-DAG:     %[[EMPTY:.+]] = tensor.empty{{.*}} : tensor<?x?xf32>

// SCALABLE-DAG:  %[[C8:.+]] = arith.constant 8 : index
// SCALABLE:      %[[VSCALE:.+]] = vector.vscale
// SCALABLE:      %[[C8_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C8]] : index

// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[INPUT]]
// CHECK-SAME:        outer_dims_perm = [0, 1] inner_dims_pos = [0, 1]
// STATIC-SAME:       inner_tiles = [7, 16]
// SCALABLE-SAME:     inner_tiles = [7, %[[C8_VSCALE]]]

// CHECK:         return %[[UNPACK]] : tensor<?x?xf32>

// RISC-V64 with V extension - matmul lowering (f32)
func.func @matmul_lowering_f32f32f32_riscv64(
    %lhs: tensor<?x?xf32, #encoding_lhs>,
    %rhs: tensor<?x?xf32, #encoding_rhs>,
    %result: tensor<?x?xf32, #encoding_res>
) -> tensor<?x?xf32, #encoding_res> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xf32, #encoding_lhs>, tensor<?x?xf32, #encoding_rhs>)
    outs(%result : tensor<?x?xf32, #encoding_res>)
    -> tensor<?x?xf32, #encoding_res>
  return %matmul : tensor<?x?xf32, #encoding_res>
}

// CHECK-LABEL: func @matmul_lowering_f32f32f32_riscv64(
// STATIC-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xf32>
// STATIC-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf32>
// STATIC-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x7x16xf32>

// SCALABLE-SAME: %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xf32>
// SCALABLE-SAME: %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x1xf32>
// SCALABLE-SAME: %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x7x?xf32>

// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
// CHECK:         return %[[MMT4D]]

// Checks that the +zvl512b flag indeed reflects on the data-tiled layout for the static case
// and does not make a difference on the scalable case.

func.func @matmul_lowering_f32f32f32_riscv64_zvl512b(
    %lhs: tensor<?x?xf32, #encoding_lhs>,
    %rhs: tensor<?x?xf32, #encoding_rhs>,
    %result: tensor<?x?xf32, #encoding_res>
) -> tensor<?x?xf32, #encoding_res> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v,+zvl512b", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xf32, #encoding_lhs>, tensor<?x?xf32, #encoding_rhs>)
    outs(%result : tensor<?x?xf32, #encoding_res>)
    -> tensor<?x?xf32, #encoding_res>
  return %matmul : tensor<?x?xf32, #encoding_res>
}

// CHECK-LABEL: func @matmul_lowering_f32f32f32_riscv64_zvl512b(
// STATIC-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xf32>

// +v implies +zvl128b, so the data-tiled layout without the flag had 16 for the N dim.
// +zvl512b therefore has 16 * (512/128) = 64

// STATIC-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x64x1xf32>
// STATIC-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x7x64xf32>

// SCALABLE-SAME: %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xf32>
// SCALABLE-SAME: %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x1xf32>
// SCALABLE-SAME: %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x7x?xf32>

// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
// CHECK:         return %[[MMT4D]]

// -----

// RISC-V32 with ukernels - uses mmt4d with default tiles

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_riscv32_ukernel(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi8, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv32-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %out = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>,
                       tensor<?x?xi8, #encoding_rhs>)
      outs(%result : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %out : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_riscv32_ukernel(
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x4xi8>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x4xi8>
// CHECK-SAME:    %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi32>
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:      ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:      outs(%[[ACC]]
// CHECK:         return %[[MMT4D]]

// -----

// RISC-V 64 + xsmtvdot + zvl256b: IME 12x16x8 tile (3x4 of 4x4x8).
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_riscv64_zvl256b(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi8, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v,+zvl256b,+xsmtvdot", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %out = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>,
                       tensor<?x?xi8, #encoding_rhs>)
      outs(%result : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %out : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_riscv64_zvl256b(
// STATIC-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x12x8xi8>
// STATIC-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x16x8xi8>
// STATIC-SAME:   %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?x12x16xi32>
// SCALABLE-SAME: %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xi8>
// SCALABLE-SAME: %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x1xi8>
// SCALABLE-SAME: %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?x7x?xi32>
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:      ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:      outs(%[[ACC]]
// CHECK:         return %[[MMT4D]]

// -----

// RISC-V 64 + xsmtvdot + zvl1024b: IME 24x32x16 tile (3x4 of 8x8x16).
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_riscv64_zvl1024b(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi8, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v,+zvl1024b,+xsmtvdot", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %out = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>,
                       tensor<?x?xi8, #encoding_rhs>)
      outs(%result : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %out : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_riscv64_zvl1024b(
// STATIC-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x24x16xi8>
// STATIC-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x32x16xi8>
// STATIC-SAME:   %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?x24x32xi32>
// SCALABLE-SAME: %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xi8>
// SCALABLE-SAME: %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x1xi8>
// SCALABLE-SAME: %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?x7x?xi32>
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:      ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:      outs(%[[ACC]]
// CHECK:         return %[[MMT4D]]

// -----

// RISC-V 64 + xsmtvdot + zvl4096b: IME 48x64x32 tile (3x4 of 16x16x32).
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_riscv64_zvl4096b(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi8, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v,+zvl4096b,+xsmtvdot", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %out = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>,
                       tensor<?x?xi8, #encoding_rhs>)
      outs(%result : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %out : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_riscv64_zvl4096b(
// STATIC-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x48x32xi8>
// STATIC-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x64x32xi8>
// STATIC-SAME:   %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?x48x64xi32>
// SCALABLE-SAME: %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xi8>
// SCALABLE-SAME: %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x1xi8>
// SCALABLE-SAME: %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?x7x?xi32>
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:      ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:      outs(%[[ACC]]
// CHECK:         return %[[MMT4D]]

// -----

// RISC-V 64 + xsmtvdot + zvl512b: VLEN not in {256, 1024, 4096}, so fall
// through to standard RVV i8 tiles. IME vmadot is not selected.
// Static: N0=vlen/8=64. Scalable: N is vscale-based (base VLEN=64).
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_riscv64_zvl512b_fallback(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi8, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v,+zvl512b,+xsmtvdot", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %out = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>,
                       tensor<?x?xi8, #encoding_rhs>)
      outs(%result : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %out : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_riscv64_zvl512b_fallback(
// STATIC-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xi8>
// STATIC-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x64x1xi8>
// STATIC-SAME:   %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?x7x64xi32>

// SCALABLE-SAME: %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x7x1xi8>
// SCALABLE-SAME: %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x1xi8>
// SCALABLE-SAME: %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?x7x?xi32>

// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:      ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:      outs(%[[ACC]]
// CHECK:         return %[[MMT4D]]

// -----

// Checks that we don't transpose M/N for the narrow-N case when scalable inner tiles are present.
// For scalable vectors, we want to keep the N dimension scalable.

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [100, 4, 500]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [100, 4, 500]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [100, 4, 500]>
func.func @matmul_lowering_narrow_n_f32f32f32_riscv64(
    %lhs: tensor<100x500xf32, #encoding_lhs>,
    %rhs: tensor<500x4xf32, #encoding_rhs>,
    %result: tensor<100x4xf32, #encoding_result>
) -> tensor<100x4xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %matmul = linalg.matmul
    ins(%lhs, %rhs : tensor<100x500xf32, #encoding_lhs>, tensor<500x4xf32, #encoding_rhs>)
    outs(%result : tensor<100x4xf32, #encoding_result>)
    -> tensor<100x4xf32, #encoding_result>
  return %matmul : tensor<100x4xf32, #encoding_result>
}

// CHECK-LABEL: func @matmul_lowering_narrow_n_f32f32f32_riscv64(
// STATIC-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<7x500x16x1xf32>
// STATIC-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x500x4x1xf32>
// STATIC-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x7x4x16xf32>

// SCALABLE-SAME: %[[ARG0:[a-zA-Z0-9]+]]: tensor<15x500x7x1xf32>
// SCALABLE-SAME: %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x500x?x1xf32>
// SCALABLE-SAME: %[[ARG2:[a-zA-Z0-9]+]]: tensor<15x?x7x?xf32>

// STATIC:         %[[MMT4D:.+]] = linalg.mmt4d
// STATIC-SAME:       ins(%[[ARG1]], %[[ARG0]] :
// STATIC-SAME:       outs(%[[ARG2]] :

// SCALABLE:       %[[MMT4D:.+]] = linalg.mmt4d
// SCALABLE-SAME:     ins(%[[ARG0]], %[[ARG1]] :
// SCALABLE-SAME:     outs(%[[ARG2]] :

// CHECK:         return %[[MMT4D]]

// -----

// RISC-V64 with V extension and zvfh - full matmul lowering (f16)

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @full_matmul_lowering_f16f16f16_riscv64(
    %lhs: tensor<?x?xf16>,
    %rhs: tensor<?x?xf16>,
    %acc: tensor<?x?xf16>,
    %m : index, %n : index, %k : index
) -> tensor<?x?xf16> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v,+zvfh", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %lhs encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #encoding_lhs>
  %1 = iree_encoding.set_encoding %rhs encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #encoding_rhs>
  %2 = iree_encoding.set_encoding %acc encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #encoding_result>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xf16, #encoding_lhs>,
                   tensor<?x?xf16, #encoding_rhs>)
      outs(%2 : tensor<?x?xf16, #encoding_result>)
      -> tensor<?x?xf16, #encoding_result>
  %4 = iree_encoding.unset_encoding %3 encoding_dims{%m, %n, %k} : tensor<?x?xf16, #encoding_result> -> tensor<?x?xf16>{%m, %n}
  return %4 : tensor<?x?xf16>
}

/// F16 on RISC-V64 with V+zvfh extension:
/// For the static case: M=7, N=16 (vlen=128), K=1
/// For the scalable case: M=7, N=8*vscale (vlen=64*vscale), K=1

// CHECK-LABEL: func @full_matmul_lowering_f16f16f16_riscv64(
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?xf16>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?xf16>
// CHECK-SAME:    %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?xf16>

// SCALABLE:      %[[C8:.+]] = arith.constant 8

/// Pack LHS: [M, K] -> [?, ?, 7, 1] (this happens before vscale computation)
// CHECK:         %[[PACK_LHS:.+]] = linalg.pack %[[LHS]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [7, 1]

// SCALABLE:      %[[VSCALE:.+]] = vector.vscale
// SCALABLE:      %[[C8_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C8]]

/// Pack RHS: [K, N] -> [?, ?, N_tile, 1] with outer_dims_perm = [1, 0]
// CHECK:         %[[PACK_RHS:.+]] = linalg.pack %[[RHS]]
// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]
// STATIC-SAME:     inner_tiles = [16, 1]
// SCALABLE-SAME:   inner_tiles = [%[[C8_VSCALE]], 1]

/// Pack ACC: [M, N] -> [?, ?, 7, N_tile]
// CHECK:         %[[PACK_ACC:.+]] = linalg.pack %[[ACC]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// STATIC-SAME:     inner_tiles = [7, 16]
// SCALABLE-SAME:   inner_tiles = [7, %[[C8_VSCALE]]]

/// The mmt4d operation
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:      ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:      outs(%[[PACK_ACC]] :

/// Unpack result: [?, ?, 7, N_tile] -> [M, N]
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[MMT4D]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// STATIC-SAME:     inner_tiles = [7, 16]
// SCALABLE-SAME:   inner_tiles = [7, %[[C8_VSCALE]]]

// CHECK:         return %[[UNPACK]]

// -----

// RISC-V64 with V extension and zvfbfwma - full matmul lowering (bf16 -> f32)

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @full_matmul_lowering_bf16bf16f32_riscv64(
    %lhs: tensor<?x?xbf16>,
    %rhs: tensor<?x?xbf16>,
    %acc: tensor<?x?xf32>,
    %m : index, %n : index, %k : index
) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v,+zvfbfwma", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %lhs encoding_dims{%m, %n, %k} : tensor<?x?xbf16> -> tensor<?x?xbf16, #encoding_lhs>
  %1 = iree_encoding.set_encoding %rhs encoding_dims{%m, %n, %k} : tensor<?x?xbf16> -> tensor<?x?xbf16, #encoding_rhs>
  %2 = iree_encoding.set_encoding %acc encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_result>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xbf16, #encoding_lhs>,
                   tensor<?x?xbf16, #encoding_rhs>)
      outs(%2 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  %4 = iree_encoding.unset_encoding %3 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #encoding_result> -> tensor<?x?xf32>{%m, %n}
  return %4 : tensor<?x?xf32>
}

/// BF16 -> F32 on RISC-V64 with V+zvfbfwma extension:
/// For the static case: M=7, N=16 (vlen=128), K=1
/// For the scalable case: M=7, N=8*vscale (vlen=64*vscale), K=1
/// The bf16 operands are LMUL=2 and the f32 accumulators EMUL=4.

// CHECK-LABEL: func @full_matmul_lowering_bf16bf16f32_riscv64(
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?xbf16>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?xbf16>
// CHECK-SAME:    %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?xf32>

// SCALABLE:      %[[C8:.+]] = arith.constant 8

/// Pack LHS: [M, K] -> [?, ?, 7, 1] (this happens before vscale computation)
// CHECK:         %[[PACK_LHS:.+]] = linalg.pack %[[LHS]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [7, 1]

// SCALABLE:      %[[VSCALE:.+]] = vector.vscale
// SCALABLE:      %[[C8_VSCALE:.+]] = arith.muli %[[VSCALE]], %[[C8]]

/// Pack RHS: [K, N] -> [?, ?, N_tile, 1] with outer_dims_perm = [1, 0]
// CHECK:         %[[PACK_RHS:.+]] = linalg.pack %[[RHS]]
// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]
// STATIC-SAME:     inner_tiles = [16, 1]
// SCALABLE-SAME:   inner_tiles = [%[[C8_VSCALE]], 1]

/// Pack ACC: [M, N] -> [?, ?, 7, N_tile]
// CHECK:         %[[PACK_ACC:.+]] = linalg.pack %[[ACC]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// STATIC-SAME:     inner_tiles = [7, 16]
// SCALABLE-SAME:   inner_tiles = [7, %[[C8_VSCALE]]]

/// The mmt4d operation
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:      ins(%[[PACK_LHS]], %[[PACK_RHS]] :
// CHECK-SAME:      outs(%[[PACK_ACC]] :

/// Unpack result: [?, ?, 7, N_tile] -> [M, N]
// CHECK:         %[[UNPACK:.+]] = linalg.unpack %[[MMT4D]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// STATIC-SAME:     inner_tiles = [7, 16]
// SCALABLE-SAME:   inner_tiles = [7, %[[C8_VSCALE]]]

// CHECK:         return %[[UNPACK]]

// -----

// RISC-V64 +v +zvl256b inner_tiled f32.

#map_it = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_it1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_it2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_it_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map_it, #map_it1, #map_it2], iteration_sizes = [?, ?, ?]>
#encoding_it_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map_it, #map_it1, #map_it2], iteration_sizes = [?, ?, ?]>
#encoding_it_res = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map_it, #map_it1, #map_it2], iteration_sizes = [?, ?, ?]>

func.func @pack_gemm_fill_dynamic_inner_tiled_rvv_vlen256(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v,+zvl256b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_it_lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_it_rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #encoding_it_res>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #encoding_it_res>)
      -> tensor<?x?xf32, #encoding_it_res>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #encoding_it_lhs>, tensor<?x?xf32, #encoding_it_rhs>)
      outs(%3 : tensor<?x?xf32, #encoding_it_res>) -> tensor<?x?xf32, #encoding_it_res>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #encoding_it_res> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @pack_gemm_fill_dynamic_inner_tiled_rvv_vlen256(
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_1x32x1_F32_F32, intrinsics_m = 6>, semantics = #iree_cpu.mma_semantics<>

// -----

#map_it_se = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_it_se1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_it_se2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_it_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map_it_se, #map_it_se1, #map_it_se2], iteration_sizes = [127, 255, ?]>
#encoding_it_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map_it_se, #map_it_se1, #map_it_se2], iteration_sizes = [127, 255, ?]>
#encoding_it_res = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map_it_se, #map_it_se1, #map_it_se2], iteration_sizes = [127, 255, ?]>
func.func @set_encoding_matmul_LHS_inner_tiled_rvv_vlen256(%arg0: tensor<127x255xf32>, %k: index) -> tensor<127x255xf32, #encoding_it_lhs> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v,+zvl256b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%k} : tensor<127x255xf32> -> tensor<127x255xf32, #encoding_it_lhs>
  return %0 : tensor<127x255xf32, #encoding_it_lhs>
}
func.func @set_encoding_matmul_RHS_inner_tiled_rvv_vlen256(%arg0: tensor<127x255xf32>, %k: index) -> tensor<127x255xf32, #encoding_it_rhs> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v,+zvl256b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%k} : tensor<127x255xf32> -> tensor<127x255xf32, #encoding_it_rhs>
  return %0 : tensor<127x255xf32, #encoding_it_rhs>
}
func.func @unset_encoding_matmul_RESULT_inner_tiled_rvv_vlen256(%arg0: tensor<127x255xf32, #encoding_it_res>, %k: index) -> tensor<127x255xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v,+zvl256b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = iree_encoding.unset_encoding %arg0 encoding_dims{%k} : tensor<127x255xf32, #encoding_it_res> -> tensor<127x255xf32>
  return %0 : tensor<127x255xf32>
}
// 127 ceildiv 6 = 22, 255 ceildiv 32 = 8
// CHECK-LABEL: func @set_encoding_matmul_LHS_inner_tiled_rvv_vlen256(
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<127x255xf32>
//       CHECK:   %[[PACK:.+]] = linalg.pack %[[INPUT]] {{.*}} inner_tiles = [6, 1]
//       CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[PACK]]
//  CHECK-SAME:     : tensor<22x255x6x1xf32> into tensor<22x255x6x1x1xf32>
//       CHECK:   return %[[EXPAND]] : tensor<22x255x6x1x1xf32>
// CHECK-LABEL: func @set_encoding_matmul_RHS_inner_tiled_rvv_vlen256(
//       CHECK:   %[[PACK_R:.+]] = linalg.pack {{.*}} inner_tiles = [32, 1]
//       CHECK:   return %[[PACK_R]] : tensor<8x127x32x1xf32>
// CHECK-LABEL: func @unset_encoding_matmul_RESULT_inner_tiled_rvv_vlen256(
//  CHECK-SAME:   %[[PACKED:[a-zA-Z0-9]+]]: tensor<22x8x6x1x32xf32>
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[PACKED]]
//  CHECK-SAME:     : tensor<22x8x6x1x32xf32> into tensor<22x8x6x32xf32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[COLLAPSE]] {{.*}} inner_tiles = [6, 32]

// -----

// inner_tiled without +zvl256b falls back to generic-scalar.
#map_g = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_g1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_g2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_g_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map_g, #map_g1, #map_g2], iteration_sizes = [?, ?, ?]>
#encoding_g_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map_g, #map_g1, #map_g2], iteration_sizes = [?, ?, ?]>
#encoding_g_res = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map_g, #map_g1, #map_g2], iteration_sizes = [?, ?, ?]>
func.func @matmul_inner_tiled_rvv_no_zvl256b(
    %lhs: tensor<?x?xf32, #encoding_g_lhs>,
    %rhs: tensor<?x?xf32, #encoding_g_rhs>,
    %acc: tensor<?x?xf32, #encoding_g_res>
) -> tensor<?x?xf32, #encoding_g_res> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv64-xyz-xyz", cpu_features="+v", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32, #encoding_g_lhs>, tensor<?x?xf32, #encoding_g_rhs>)
      outs(%acc : tensor<?x?xf32, #encoding_g_res>) -> tensor<?x?xf32, #encoding_g_res>
  return %0 : tensor<?x?xf32, #encoding_g_res>
}
// CHECK-LABEL: func @matmul_inner_tiled_rvv_no_zvl256b(
//       CHECK:   %[[INNER_G:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:     kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_GENERIC_SCALAR_1x1x1_REG16, intrinsics_m = 3, intrinsics_n = 3, lhs_type = f32, rhs_type = f32, acc_type = f32>
