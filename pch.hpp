#include "tensor/host_buffer/functions.hpp"
#include "tensor/types.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include <cstddef>
#include <tt_eager/tensor/tensor.hpp>
#include <ttnn/core.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/device.hpp>
#include <tt_dnn/op_library/fully_connected/fully_connected_op.hpp>
#include <tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp>
#include <tt_eager/tensor/tensor.hpp>
#include <tt_dnn/op_library/transpose/transpose_op.hpp>

#include "common/bfloat16.hpp"
#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/matmul.hpp>
#include <tt_dnn/op_library/update_cache/update_cache_op.hpp>

#include <vector>
#include <iostream>