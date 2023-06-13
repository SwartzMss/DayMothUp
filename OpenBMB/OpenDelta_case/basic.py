from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from opendelta import AutoDeltaModel, AutoDeltaConfig
from opendelta import Visualization



t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
t5_tokenizer = AutoTokenizer.from_pretrained("t5-large")
# 可视化NN
Visualization(t5_model.encoder).structure_graph()
Visualization(t5_model.decoder).structure_graph()

'''
在transformers库中，encode方法和__call__方法（即直接调用tokenizer对象）实际上有一些区别。
1. encode方法将文本转换为一系列的token ID，但不会返回任何额外的信息。此外，它只接受单个字符串作为输入。
2. __call__方法（即直接调用tokenizer对象）不仅返回token ID，还返回一些额外的信息，例如attention mask。
此外，它可以接受一对字符串作为输入，这在处理两个文本序列的任务（例如问答或序列分类）时非常有用。
'''
inputs_ids = t5_tokenizer.encode("Is Harry Poter wrtten by JKrowling", return_tensors="pt")
print(f"inputs_ids[0] = {inputs_ids[0]}")
inputs  = t5_tokenizer(["Is Harry Potter written by Rowling"], return_tensors="pt")
print(f"inputs = {inputs}")
'''
inputs_ids[0] = tensor([  27,    7, 8929, 7995,   49,    3,  210,   52,   17,  324,   57,  446,
         439, 3623,  697,    1])
inputs = {'input_ids': tensor([[  27,    7, 8929, 7995,   49,    3,  210,   52,   17,  324,   57,  446,
          439, 3623,  697,    1]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
'''


answers_ids = t5_model.generate(input_ids=inputs.input_ids, max_length=20)
print(f"answers_ids = {answers_ids}")
'''
skip_special_tokens 默认是false,会有一些特殊的标记
input_text = <pad><extra_id_0>.<extra_id_1> Is Harry Potter written by Rowling.<extra_id_2>.<extra_id_3>
'''
input_text = t5_tokenizer.decode(answers_ids[0], skip_special_tokens = True)
print(f"input_text = {input_text}")


'''
LoraConfig {
  "lora_alpha": 16,
  "lora_dropout": 0.0,
  "lora_r": 8,
  "opendelta_version": "0.3.0",
  "transformers_version": "4.30.0.dev0"
}
'''
delta_config1 = AutoDeltaConfig.from_dict({"delta_type":"lora"})
delta1 = AutoDeltaModel.from_config(delta_config1, backbone_model=t5_model)
delta1.log()
delta1.detach()
'''
│   │           │   │   ├── q,v(Linear) weight:[1024, 1024]
│   │           │   │   │   └── lora (LowRankLinear) lora_A:[8, 1024] lora_B:[1024, 8]
'''

'''
LoraConfig {
  "lora_alpha": 16,
  "lora_dropout": 0.0,
  "lora_r": 5,
  "modified_modules": [
    "[r]decoder.*((20)|(21)|(22)|(23)).*DenseReluDense\\.wi"
  ],
  "opendelta_version": "0.3.0",
  "transformers_version": "4.30.0.dev0"
}
'''

delta_config2 = AutoDeltaConfig.from_dict({"delta_type":"lora", "modified_modules":["[r]decoder.*((20)|(21)|(22)|(23)).*DenseReluDense\.wi"], "lora_r":5})
delta2 = AutoDeltaModel.from_config(delta_config2, backbone_model=t5_model)
delta2.freeze_module()
delta2.log()
'''
    │   └── 20-23(T5Block)
    │       └── layer (ModuleList)
    │           ├── 0 (T5LayerSelfAttention)
    │           │   ├── SelfAttention (T5Attention)
    │           │   │   ├── q,v(Linear) weight:[1024, 1024]
    │           │   │   │   └── lora (LowRankLinear) lora_A:[8, 1024] lora_B:[1024, 8]
    │           │   │   └── k,o(Linear) weight:[1024, 1024]
    │           │   └── layer_norm (T5LayerNorm) weight:[1024]
    │           ├── 1 (T5LayerCrossAttention)
    │           │   ├── EncDecAttention (T5Attention)
    │           │   │   └── q,k,v,o(Linear) weight:[1024, 1024]
    │           │   └── layer_norm (T5LayerNorm) weight:[1024]
    │           └── 2 (T5LayerFF)
    │               ├── DenseReluDense (T5DenseActDense)
    │               │   ├── wi (Linear) weight:[4096, 1024]
    │               │   │   └── lora (LowRankLinear) lora_A:[5, 1024] lora_B:[4096, 5]
'''

# add optimizer as normal
from transformers import AdamW
optimizer = AdamW(t5_model.parameters(), lr=3e-3)

# inspect_optimizer
from opendelta.utils.inspect import inspect_optimizer_statistics
inspect_optimizer_statistics(optimizer)