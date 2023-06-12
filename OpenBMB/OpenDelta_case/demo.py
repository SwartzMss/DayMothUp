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