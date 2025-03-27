#include "cfc.h"
#include "../../functional.h"
namespace nt {
namespace ncps {

TensorGrad CfC::forward(TensorGrad input, const TensorGrad &hx, Tensor timespan,
                        TensorGrad &hx_out) {
    DType dtype = input.dtype;
    bool is_batched = input.dims() == 3;
    int64_t batch_dim = (this->batch_first) ? 0 : 1;
    int64_t seq_dim = (this->batch_first) ? 1 : 0;
    if (!is_batched) {
        input = input.unsqueeze(batch_dim);
        if (!timespan.is_null()) {
            timespan = timespan.unsqueeze(batch_dim);
        }
    }
    int64_t batch_size = input.shape()[batch_dim];
    int64_t seq_len = input.shape()[seq_dim];

    TensorGrad h_state(nullptr);
    TensorGrad c_state(nullptr);

    if (hx.is_null()) {
        h_state = functional::zeros({batch_size, this->state_size}, dtype);
        if (this->use_mixed) {
            c_state = functional::zeros({batch_size, this->state_size}, dtype);
        }
    } else {
        if (this->use_mixed) {
            utils::THROW_EXCEPTION(
                hx.dtype == DType::TensorObj && hx.numel() == 2,
                "Running a CfC with mixed_memory=true, requires a tensor "
                "obj of 2 tensors (h0, c0) to be passed as state but got a "
                "tensor of dtype $ instead",
                hx.dtype);
            h_state = hx[0];
            c_state = hx[1];
        } else {
            utils::THROW_EXCEPTION(
                hx.dtype == dtype,
                "Expected when Running a CfC with mixed_memory=false, a "
                "dtype same as input of $ but fot $",
                dtype, hx.dtype);
            h_state = hx;
        }
        if (is_batched) {
            utils::THROW_EXCEPTION(h_state.dims() == 2,
                                   "For batched 2-D input, hx and cx "
                                   "should also be 2-D but got $-D",
                                   h_state.dims());
        } else {
            utils::THROW_EXCEPTION(h_state.dims() == 1,
                                   "For unbatched 1-D input, hx and cx "
                                   "should also be 1-D but got $-D",
                                   h_state.dims());
            h_state = h_state.unsqueeze(0);
            if (!c_state.is_null()) {
                c_state = c_state.unsqueeze(0);
            }
        }
    }
    TensorGrad readout(nullptr);
    TensorGrad inputs_split = input.transpose(0, seq_dim);
    inputs_split = inputs_split.split_axis(0);

    Tensor timespans = (timespan.is_null()) ? Tensor(Scalar(1.0))
                                            : timespan.split_axis(seq_dim);
    std::vector<TensorGrad> output_sequence(seq_len,
                                            TensorGrad(Tensor::Null()));
    for (int64_t t = 0; t < seq_len; ++t) {
        Tensor ts =
            timespan.is_null() ? timespans : timespans[t].item<Tensor>();

        TensorGrad inputs = inputs_split[t];
        if (this->use_mixed) {
            std::tie(h_state, c_state) =
                get<2>(this->lstm(inputs, h_state, c_state));
        }
        TensorGrad h_out = this->rnn_cell(inputs, h_state, ts, h_state);
        output_sequence[t] = h_out;
    }
    if (this->return_sequences) {
        readout = functional::stack(output_sequence, seq_dim);
    } else {
        readout = this->fc(output_sequence.back());
    }
    if (!is_batched) {
        readout = readout.squeeze(batch_dim);
        hx_out = this->use_mixed ? functional::list(h_state[0], c_state[0])
                                 : h_state[0];
    } else {
        hx_out = this->use_mixed
                     ? functional::list(std::move(h_state), std::move(c_state))
                     : std::move(h_state);
    }
    return std::move(readout);
}

} // namespace ncps
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::ncps::CfC, nt__ncps__CfC, input_size,
                               proj_size, batch_first, return_sequences,
                               wired_mode, rnn_cell, lstm, fc, state_size,
                               output_size, use_mixed)
