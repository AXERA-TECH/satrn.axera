import torch
import axengine as axe
import numpy as np
import math

def _get_source_mask(src_seq, valid_ratios) -> torch.Tensor:
    """Generate mask for source sequence.

    Args:
        src_seq (torch.Tensor): Image sequence. Shape :math:`(N, T, C)`.
        valid_ratios (list[float]): The valid ratio of input image. For
            example, if the width of the original image is w1 and the width
            after padding is w2, then valid_ratio = w1/w2. Source mask is
            used to cover the area of the padding region.

    Returns:
        Tensor or None: Source mask. Shape :math:`(N, T)`. The region of
        padding area are False, and the rest are True.
    """

    N, T, _ = src_seq.size()
    mask = None
    if len(valid_ratios) > 0:
        mask = src_seq.new_zeros((N, T), device=src_seq.device)
        for i, valid_ratio in enumerate(valid_ratios):
            valid_width = min(T, math.ceil(T * valid_ratio))
            mask[i, :valid_width] = 1

    return mask

onnx_bb_encoder = axe.InferenceSession("backbone_encoder.axmodel")
onnx_decoder = axe.InferenceSession("decoder.axmodel")

input_image  = torch.tensor(np.load('input_tensor/input_image.npy'))
out_enc = onnx_bb_encoder.run(["output"], {"input": np.array(input_image.cpu())})[0]
out_enc = torch.tensor(out_enc)
data_samples = None

N = out_enc.size(0)

init_target_seq  = torch.tensor(np.load('input_tensor/init_target_seq.npy')).to(torch.int32)
outputs = []
max_seq_len = 25
for step in range(0, max_seq_len):
    valid_ratios = [1.0 for _ in range(out_enc.size(0))]
    if data_samples is not None:
        valid_ratios = []
        for data_sample in data_samples:
            valid_ratios.append(data_sample.get('valid_ratio'))
    
    src_mask = _get_source_mask(out_enc, valid_ratios)
    # step_result = model_decoder(init_target_seq,out_enc,src_mask,step)
    step_result = onnx_decoder.run(["output"],{'init_target_seq':np.array(init_target_seq),
                                            'out_enc':np.array(out_enc),
                                            'src_mask':np.array(src_mask),
                                            'step':np.array([step]).astype(np.int32)})[0][0]
    step_result = torch.tensor(step_result)
    outputs.append(step_result)
    _, step_max_index = torch.max(step_result, dim=-1)
    init_target_seq[:, step + 1] = step_max_index
outputs = torch.stack(outputs, dim=1)
np.save('output_tensor/outputs.npy',outputs)

    
    
    
    
    