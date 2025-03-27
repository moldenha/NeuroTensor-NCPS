#ifndef _NT_LAYERS_NCPS_LSTM_CELL_H_
#define _NT_LAYERS_NCPS_LSTM_CELL_H_

#include "../Layer.h"
#include "../Module.h"
#include "../layers.h"

namespace nt {
namespace ncps {

class LSTMCell : public Module {
    int64_t input_size, hidden_size;
    void init_weights(); 

  public:
    Layer input_map, recurrent_map;
    LSTMCell(int64_t input_size, int64_t hidden_size);

    TensorGrad forward(const TensorGrad &inputs, const TensorGrad &output_state,
                       const TensorGrad &cell_state); 
};

} // namespace ncps
} // namespace nt


#endif //_NT_LAYERS_NCPS_LSTM_CELL_H_
