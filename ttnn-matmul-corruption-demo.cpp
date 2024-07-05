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

static tt::tt_metal::Tensor make_random_tensor(tt::tt_metal::Shape s)
{
    static int seed = 42;
     auto b = tt::tt_metal::owned_buffer::create(
        create_random_vector_of_bfloat16_native(
        s[0] * s[1] * s[2] * s[3] * 2
            , 2, seed++, -1));
    tt::tt_metal::Tensor t(OwnedStorage{std::move(b)}, s
        , tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::Layout::ROW_MAJOR);
    return tt::tt_metal::tilize_with_zero_padding(t.to(AutoFormat::GetDefaultDevice()));
}

int main()
{
    auto device = &ttnn::open_device(0);
    AutoFormat::SetDefaultDevice(device);
    ttnn::enable_program_cache(*device); // Program cache HAS to be enabled

    auto m1 = make_random_tensor(tt::tt_metal::Shape{1, 10, 32, 256});
    auto m2 = make_random_tensor(tt::tt_metal::Shape{1, 20, 32, 256});

    auto m3 = make_random_tensor(tt::tt_metal::Shape{1, 1, 32, 32}); // m3 must be assigned with something
    // The transpose() in the following line must be there. Directly matmul a 1x10x32x256 by 1x20x256x32 does not trigger the bug
    m3 = ttnn::operations::matmul::matmul(m2, tt::tt_metal::transpose(m1, -2, -1), std::nullopt);

    // This won't trigger the bug
    // auto m3 = ttnn::operations::matmul::matmul(m2, tt::tt_metal::transpose(m1, -2, -1), std::nullopt);

    auto& q = m3;
    std::vector<bfloat16> buf(q.shape().volume());
    tt::tt_metal::memcpy(buf.data(), q);
    std::cout << std::endl;
    for(size_t i = 0; i < buf.size(); i++) {
        if(i % 32 == 0)
            std::cout << std::endl;
        if(i % 1024 == 0)
            std::cout << std::endl;
        std::cout << buf[i].to_float() << " ";

        // detect NaN and corrupted values
        if((std::isnan(buf[i].to_float()) == true || std::abs(buf[i].to_float()) > 100)) {
            std::cerr << "NaN or corrupted value detected at index " << i << std::endl;
            abort();
        }
    }

    std::cout << std::endl;


    device->close();
}