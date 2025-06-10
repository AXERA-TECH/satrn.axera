from mmocr.apis import MMOCRInferencer
import torch
import torch.nn as nn


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


infer = MMOCRInferencer(rec='satrn')
model = infer.textrec_inferencer.model
model.eval()

# backbone+encoder
input_image = torch.randn((1, 3, 32, 100), device='cuda', dtype=torch.float32)
model_backbone_encoder = BackboneEncoderOnly(model)
model_input = (input_image)

torch.onnx.export(
    model_backbone_encoder,
    model_input,
    './onnx/satrn_backbone_encoder.onnx',
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
)
print('export backbone+encoder finish!')

# decoder
model_decoder = DecoderOnly(model)
model_decoder.eval()
init_target_seq = torch.randint(low=1,high=91,size=(1, 26), device='cuda', dtype=torch.long)
out_enc = torch.randn((1, 200, 512), device='cuda', dtype=torch.float32)
src_mask = torch.ones((1,200), device='cuda', dtype=torch.float32)
step = torch.randint(low=1,high=21,size=(1,), device='cuda', dtype=torch.long)
model_input = (init_target_seq,out_enc,src_mask,step)
output = model_decoder(init_target_seq,out_enc,src_mask,step)


torch.onnx.export(
    model_decoder,
    model_input,
    './onnx/satrn_decoder.onnx',
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['init_target_seq','out_enc','src_mask','step'],
    output_names=['output'],
)
print('export decoder finish!')