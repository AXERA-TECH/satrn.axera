from mmocr.apis import MMOCRInferencer
from mmocr.apis.inferencers.base_mmocr_inferencer import BaseMMOCRInferencer
import torch
from rich.progress import track
import torch.nn as nn
import onnxruntime as ort
import numpy as np

onnx_bb_encoder = ort.InferenceSession("onnx/satrn_backbone_encoder.onnx")
onnx_decoder = ort.InferenceSession("onnx/satrn_decoder_sim.onnx")


class BackboneEncoderOnly(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # 保留 backbone 和 encoder
        self.backbone = original_model.backbone
        self.encoder = original_model.encoder
        
    def forward(self, x):
        x = self.backbone(x)
        return self.encoder(x)


class DecoderOnly(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # 保留 backbone 和 encoder
        original_decoder = original_model.decoder
        # self._attention = original_decoder._attention
        self.classifier = original_decoder.classifier
        self.trg_word_emb = original_decoder.trg_word_emb
        self.position_enc = original_decoder.position_enc
        self._get_target_mask = original_decoder._get_target_mask
        self.dropout = original_decoder.dropout
        self.layer_stack = original_decoder.layer_stack
        self.layer_norm = original_decoder.layer_norm
        self._get_source_mask = original_decoder._get_source_mask
        self.postprocessor = original_decoder.postprocessor
        self.start_idx = 90
        self.padding_idx = 91
        self.max_seq_len = 25
        self.softmax = nn.Softmax(dim=-1)
        
        
        
    def forward(self, trg_seq,src,src_mask,step):
        # decoder_output = self._attention(init_target_seq, out_enc, src_mask=src_mask)
        trg_embedding = self.trg_word_emb(trg_seq)
        trg_pos_encoded = self.position_enc(trg_embedding)
        trg_mask = self._get_target_mask(trg_seq)
        tgt_seq = self.dropout(trg_pos_encoded)

        output = tgt_seq
        for dec_layer in self.layer_stack:
            output = dec_layer(
                output,
                src,
                self_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
        output = self.layer_norm(output)
        # bsz * seq_len * C
        step_result = self.classifier(output[:, step, :])
        return step_result



def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    对 uint8 张量进行标准化处理
    参数:
        tensor: 输入张量，形状为 [3, 32, 100]，数据类型为 uint8
    返回:
        标准化后的张量，形状不变，数据类型为 float32
    """
    # 检查输入张量的形状和数据类型
    assert tensor.shape == (3, 32, 100), "输入张量形状必须为 [3, 32, 100]"
    assert tensor.dtype == torch.uint8, "输入张量数据类型必须为 uint8"
    
    # 转换为 float32 类型
    tensor = tensor.float()
    
    # 定义标准化参数（RGB 通道顺序）
    mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1)
    
    # 执行标准化：(x - mean) / std
    normalized_tensor = (tensor - mean) / std
    
    return normalized_tensor


infer = MMOCRInferencer(rec='satrn')
model = infer.textrec_inferencer.model
model.eval()
model.cpu()
input_path = 'mmor_demo/demo/demo_text_recog.jpg'
ori_inputs = infer._inputs_to_list([input_path])
base = BaseMMOCRInferencer(model='satrn')
chunked_inputs = base._get_chunk_data(ori_inputs, 1)
for ori_inputs in track(chunked_inputs, description='Inference'):
    input = ori_inputs[0][1]
    input_img = input['inputs']
    input_image = normalize_tensor(input_img).unsqueeze(0)
    input_sample = input['data_samples']
    
    # backbone+encoder
    model_backbone_encoder = BackboneEncoderOnly(model)
    model_decoder = DecoderOnly(model)
    
    # out_enc = model_backbone_encoder(input_image)
    out_enc = onnx_bb_encoder.run(None, {"input": np.array(input_image.cpu())})[0]
    out_enc = torch.tensor(out_enc)
    data_samples = None
    
    N = out_enc.size(0)
    init_target_seq = torch.full((N, model_decoder.max_seq_len + 1),
                                model_decoder.padding_idx,
                                device=out_enc.device,
                                dtype=torch.long)
# bsz * seq_len
    init_target_seq[:, 0] = model_decoder.start_idx

    outputs = []
    for step in range(0, model_decoder.max_seq_len):
        valid_ratios = [1.0 for _ in range(out_enc.size(0))]
        if data_samples is not None:
            valid_ratios = []
            for data_sample in data_samples:
                valid_ratios.append(data_sample.get('valid_ratio'))
        
        src_mask = model_decoder._get_source_mask(out_enc, valid_ratios)
        # step_result = model_decoder(init_target_seq,out_enc,src_mask,step)
        step_result = onnx_decoder.run(None,{'init_target_seq':np.array(init_target_seq),
                                             'out_enc':np.array(out_enc),
                                             'src_mask':np.array(src_mask),
                                             'step':np.array([step])})[0][0]
        step_result = torch.tensor(step_result)
        outputs.append(step_result)
        _, step_max_index = torch.max(step_result, dim=-1)
        init_target_seq[:, step + 1] = step_max_index
    outputs = torch.stack(outputs, dim=1)
    out_dec = model_decoder.softmax(outputs)
    output = model_decoder.postprocessor(out_dec, [input_sample])
    outstr = output[0].pred_text.item
    outscore = output[0].pred_text.score
    
    print('pred_text:',outstr)
    print('score:',outscore)
    
    
    
    
    