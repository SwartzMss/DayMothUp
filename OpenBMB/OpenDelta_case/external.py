from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-base")


from bigmodelvis import Visualization
'''
  (model): BartModel(
    (shared): Embedding(50265, 768, padding_idx=1)
    (encoder): BartEncoder(
      (embed_tokens): Embedding(50265, 768, padding_idx=1)
      (embed_positions): BartLearnedPositionalEmbedding(1026, 768)
      (layers): ModuleList(
        (0-5): 6 x BartEncoderLayer(
          (self_attn): BartAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (activation_fn): GELUActivation()
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layernorm_embedding): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
'''
print(model)
print("--------------before modify---------------")
Visualization(model).structure_graph()
"""
白色部分是模块的名称。
绿色部分是模块的类型。
蓝色部分是可调参数，即需要梯度计算的参数。
灰色部分是冻结的参数，即不需要梯度计算的参数。
红色部分是重复的结构，因此是折叠的。
紫色部分是插入到主干模型中的增量参数
"""

from opendelta import LoraModel
delta_model = LoraModel(backbone_model=model, modified_modules=['fc2'])
print("--------------after modify---------------")
delta_model.log()


delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
print("------------after freeze-------------------")
delta_model.log()
# set_state_dict=True将告诉该方法将backbone_model的state_dict更改为只维护可训练的部分




'''
       │       ├── fc1 (Linear) weight:[3072, 768] bias:[3072]
│       │       └── fc2 (Linear) weight:[768, 3072] bias:[768]
│       │           └── lora (LowRankLinear) lora_A:[8, 3072] lora_B:[768, 8]

我重新理解一下 其实是LORA_B*LORA_A*fc2的一个输出？,这样的话 其实也就实现了一个降维的操作
换成矩阵运算的话[768,8]*[8,3072]*[3072] = [728,3072]*[3072] = [728]
'''