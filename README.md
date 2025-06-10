# satrn.axera
Demo for [satrn](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/satrn/README.md) on Axera device.

## 支持平台
- [x] AX650N
- [ ] AX630C

### 环境准备
按照mmocr原repo中的[installation](https://github.com/open-mmlab/mmocr/tree/main?tab=readme-ov-file#installation)安装环境


### 导出模型(PyTorch -> ONNX)
1. 导出前准备

    修改nrtr_decoder.py中的[L232-L234](https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/decoders/nrtr_decoder.py#L232)
    固定动态参数valid_ratios=1

    ```
            # valid_ratios = []
            # for data_sample in data_samples:
            #     valid_ratios.append(data_sample.get('valid_ratio'))

            valid_ratios = [1.0 for _ in range(out_enc.size(0))]
            if data_samples is not None:
                valid_ratios = []
                for data_sample in data_samples:
                    valid_ratios.append(data_sample.get('valid_ratio'))
    ```


2. 模型导出
    ```
    python export_onnx.py
    ```
    导出成功后会生成两个onnx模型:
    - backbone and image encoder: satrn_backbone_encoder.onnx
    - encoder: satrn_decoder.onnx

3. Simplify ONNX：需要对decoder进行Simplify
    ```
    pip install onnx-simplifier
    python -m onnxsim onnx/satrn_decoder.onnx onnx/satrn_decoder_sim.onnx
    ```


#### 转换模型(ONNX -> Axera)
使用模型转换工具 Pulsar2 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 .axmodel，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 Pulsar2 build 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考[AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

#### 量化数据集准备
此处随机生成数据，仅用作demo，建议使用实际参与训练的数据
- backbone_encoder数据:

    下载[dataset_v04.zip](https://github.com/user-attachments/files/20480889/dataset_v04.zip)或自行准备

- decoder数据:
    ```
    python gen_random_decoder_cali_data.py
    ```
最终得到两个数据集：

\- cali_data/dataset_v04.zip

\- cali_data/decoder_data

#### 模型编译
修改配置文件: 检查config.json 中 calibration_dataset 字段，将该字段配置的路径改为上一步准备的量化数据集存放路径

此处两个模型精度均配置为U16

在编译环境中，执行pulsar2 build参考命令：
```
# backbone_encoder
pulsar2 build --config build_config/satrn_backbone_encoder_config.json --input onnx/satrn_backbone_encoder.onnx --output_dir build_output/backbone_encoder --output_name backbone_encoder.axmodel

# decoder
pulsar2 build --config build_config/satrn_decoder.json --input onnx/satrn_decoder_sim.onnx --output_dir build_output/decoder --output_name decoder.axmodel
```



编译完成后得到两个axmodel模型：


\- backbone_encoder.axmodel

\- decoder.axmodel


### Python API 运行


#### onnx运行demo

```
python run_onnx.py
```

输入图像

![](mmor_demo/demo/demo_text_recog.jpg)


输出内容
```shell
pred_text: STAR
score: [0.9384030103683472, 0.9574987292289734, 0.9993689656257629, 0.9994958639144897]
```

#### axmodel(待验证)
```
python run_axmodel.py
```

输出内容
```shell
pred_text: STAR
score: [0.9384030103683472, 0.9574987292289734, 0.9993689656257629, 0.9994958639144897]
```


## 技术讨论

- Github issues
- QQ 群: 139953715