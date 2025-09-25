Training GPT2 Chinese from zero to hero
==

1.Description:
---
从头训练一个82M的中文GPT2模型，使用BERT的Tokenizer.中文语料采用红楼梦小说的部分章节，大小约3120KB。训练15个周期，batchsize=8。最终可以续写10句以上的红楼梦小说。

2.Start:
----
(1)***environment***

首先，我们下载依赖。
```bash
pip install -r requirements.txt
```

(2)***dataset***

准备中文语料，放置在./data/文件夹下，将语料由.txt文件更改为input.json文件
红楼梦小说语料来源于https://zh.z1917.ru/
········
按照参考样例./train.json更改input.json文件格式,由于数据集内容为原始的小说内容，包含着大量的非法字符和json读取不支持的控制字符，因此我们对原始数据集文件进行处理，去除其中非法字符，生成预处理好的数据集文件train.json。
```bash
python clr_ctrl.py
```

(3)***Model***

在model_config 定义初始GPT-2模型的超参数配置，
- "initializer_range": 0.02 ： 定义了模型参数（如权重矩阵）在初始化时的标准差，权重会在均值为0，标准差为0.02的正态分布中进行随机初始化。
- "layer_norm_epsilon": 1e-05 ： 用于层归一化的常数，用于避免在归一化过程中出现除以零的情况。设置值为1e-05，用于稳定训练。
- "n_ctx": 1024 ： 表示模型上下文窗口的大小，GPT-2 在生成文本时会考虑的最大序列长度。最大长度设为1024，即模型一次最多能处理1024个 token。
- "n_embd": 768 ： 表示每个token的嵌入维度大小，即模型中词向量的维度。设置为768，即每个词汇的表示向量是768维的。
- "n_head": 12 ： 表示自注意力机制中的注意力头的数量。设置为12，即模型的多头注意力机制中有12个独立的头。
- "n_layer": 10 ： 表示 Transformer 编码器中的层数。在这里，设置为 12，即模型有 12 层堆叠的 Transformer 块。
- "n_positions": 1024 ： 表示模型可以处理的最大位置索引，即序列中的最大位置数。最大位置数为 1024，和 n_ctx一致，表示模型最多能处理1024个位置的token。
- "vocab_size": 13317 ： 表示词汇表的大小，即模型可以识别和生成的词汇数量。在这里，词汇表大小为 21128，表示该模型可以处理的词汇量为21128个不同的 token。


(4)***Training***
python train.py --model_config/model_config_small.json --tokenized_path data/tokenized/ --tokenizer_path cache/vocab_small.txt --raw_data_path data/train.json

现在，我们可以使用我们处理好的数据集来训练我们的初始gpt2模型，使用如下命令：
```bash
python train.py   --model_config config/model_config_small.json   --tokenized_data_path data/tokenized/   --tokenizer_path cache/vocab_small.txt   --raw_data_path data/train.json   --epochs 15   --log_step 200   --stride 512   --output_dir model/   --device 0,1   --num_pieces 100   --raw
```

在这个过程中，我们可以看到命令窗口打印出模型的config文件，定义了模型的结构；同时也打印出了模型的参数量，为81894144，约82M

Print Model config
config:
{
  "attn_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "finetuning_task": null,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 10,
  "n_positions": 1024,
  "num_labels": 1,
  "output_attentions": false,
  "output_hidden_states": false,
  "output_past": true,
  "pruned_heads": {},
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "torchscript": false,
  "use_bfloat16": false,
  "vocab_size": 13317
}
number of parameters: 81894144

训练过程中，每个epoch对应的模型都将存储在./model/目录下，最终训练好的模型将存储在./model/final_model/路径中。

(5)***Generate***

现在，我们可以使用我们用目标语料训练生成的模型来进行文字生成，使用如下命令：
```bash
python generate.py   --device 1   --length 1000   --tokenizer_path cache/vocab_small.txt   --model_path model/final_model   --prefix "[CLS]黛玉咳嗽一声一声"   --topp 1   --temperature 1.0 --save_samples --save_samples_path ./mnt/
```

3.Result
--
最终会生成10个文字样本，存储在./mnt/目录下，其中之一如下：

======================================== SAMPLE 1 ========================================

黛玉咳嗽一声一声，只得得得一声儿。紫鹃哭道：“好姐姐，你们姑娘们姑娘们姑娘们说的话，说话。”紫鹃道：“不是宝二爷还要回去的，你们只有什么，我就是和姑娘说话，姑娘们姑娘也是姑娘们姑娘听着呢，如今听见了呢。”紫鹃听了，便说道：“我也不是宝二爷和二爷这里是什么，只要打发我们说的了！”一声音未出话，一面一面说不住，一面一面又嗽了，一叠声，一面紫鹃，连忙忙的一面叫道：“你听什么话了，你们这会子叫我说呢。”紫鹃道：“姑娘这话，我才是为什么来，你们只管去叫我们家去。”紫鹃道：“姑娘这样人叫我们这里就说，说了，叫我们姑娘们就来瞧罢。”黛玉点头儿，一面说：“姑娘和我们都在屋里听见了。”紫鹃听了，忙忙出来。那里间来说“紫鹃姑娘这般光景，还要紧大了不叫我们，你们说什么？我们两个人家又说我们只是谁知道了。”紫鹃道：“我也没什么，我们只不是这话说什么，不得说什么话了。”雪雁连忙答，又叫人来说着出去。雪雁出来，紫鹃雪雁已进屋门，那边坐在那里间，黛玉的那眼，只见黛玉，便说道：“姑娘今日好了，叫紫鹃姐们姑娘坐着罢，我们姑娘坐着罢。”紫鹃忙叫他们也问道：“你们姑娘说什么？”雪雁进来。黛玉也问道：“你们两个屋里坐坐罢，你们两天天天天才听。”翠缕道：“你们姑娘怎么不得早起，叫我去了？”黛玉道：“你听见宝二爷没说话，我们说什么没听见过，又是什么话？”雪雁点点点点点头儿道：“我们姑娘说什么。”雪雁道：“我也不过来？”黛玉道：“你这话来了，你们二爷回来瞧姑娘的。”说着，回来。宝玉便问道：“我们家，也不过来了，又说起来瞧姑娘。”黛玉道：“我们宝二爷已经，也没有见过去了。”说着，便走，便问道：“你们说话。”雪雁道：“宝二爷叫我们在屋里坐坐着，还没有？”黛玉道：“没有？”宝玉道：“宝二爷没听得。”宝玉道：“宝二爷今夜没说了，没见说的时候，也没有。”黛玉又问道：“那里说什么，还没有？”黛玉道：“宝姐姐说什么。我们宝二爷叫他们没有？”黛玉道：“那边坐坐坐坐坐坐罢？”黛玉道：“我们，看见了一天要坐，也不吃，只不去。”宝玉便叫雪雁告诉雪雁进来说了。这里宝玉道：“那些什么？”黛玉道：“别叫我说，还在外头儿坐坐着罢，不吃饭。”黛玉道：“那里头里坐着，我也没有。”又把心里也不理他，只得一口，便叫雪雁告诉袭人，便告诉他。黛玉点点儿一口，便问：“他们家，怎么？”宝玉道：“你去了。”黛玉道：“你们在外头上屋里坐坐下罢。”紫鹃也不言，也不敢言语了，只得
==========================================================================================
