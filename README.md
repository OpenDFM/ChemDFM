# ChemDFM: A Large Language Foundation Model for Chemistry

<div style="text-align: center;">
    <a href="https://doi.org/10.5281/zenodo.14913412"><img src="https://zenodo.org/badge/768631984.svg" alt="DOI"></a>
</div>

![Main Image](https://github.com/OpenDFM/ChemDFM/raw/main/docs/static/images/main.png)

ChemDFM is the pioneering open-sourced dialogue foundation model for Chemistry and molecular science, which is built based on LLaMa-13B. ChemDFM outperforms the open-sourced LLMs in all the typical tasks of chemistry, and even reaches comparable or higher performances to GPT-4. For more details, please refer to [our paper](https://arxiv.org/abs/2401.14818).

## News

* **2025-04-16**: Our paper is accepted by *Cell Report Physical Science*. The published version can be accessed [HERE](https://www.sciencedirect.com/science/article/pii/S2666386425001225)
* **2024-11-09**: [ChemDFM-v1.5-8B](https://huggingface.co/OpenDFM/ChemDFM-v1.5-8B) is released! We implemented our domain pre-training and instruction tuning procedure on a stronger base model LLaMA-3-8B.
* **2024-06-13**: The results on the comprehensive science benchmark [SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval) show that "ChemDFM emerged as one of the top open-source models by continuing pre-training and fine-tuning on a vast corpus of scientific literature".
* **2024-04-17**: The evaluation data (including instructions) we used in our paper is released on [GitHub](https://github.com/OpenDFM/ChemDFM)
* **2024-03-12**: The parameter of [ChemDFM-v1.0-13B](https://huggingface.co/OpenDFM/ChemDFM-v1.0-13B) is open-sourced!
* **2024-01-26**: The paper of ChemDFM-13B is released on arXiv: [ChemDFM: Dialogue Foundation Model for Chemistry](https://arxiv.org/abs/2401.14818)

## Usage Details

The model parameters and online demo of ChemDFM will be up soon!

### local inference

To load and run ChemDFM locally, here is an example:

```python
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

model_name_or_id = "OpenDFM/ChemDFM-v1.5-8B"
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_id)
model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto")

input_text = "Can you please give detailed descriptions of the molecule below?\nCl.O=C1c2c(O)cccc2-c2nn(CCNCCO)c3ccc(NCCNCCO)c1c23"
input_text = f"[Round 0]\nHuman: {input_text}\nAssistant:"

inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
generation_config = GenerationConfig(
    do_sample=True,
    top_k=20,
    top_p=0.9,
    temperature=0.9,
    max_new_tokens=1024,
    repetition_penalty=1.05,
    eos_token_id=tokenizer.eos_token_id
)

outputs = model.generate(**inputs, generation_config=generation_config)
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(input_text):]
print(generated_text.strip())
```

### input format

To get better responses, we recommend to preprocess your input and history with the dialogue templates which are used during instruction tuning of ChemDFM. Specifically, for an input queries
```python
{'current_query': current_query, 'history': [(query1, answer1), (query2, answer2), ...]}
```
, you can use the following code to preprocess the input and history:
```python
def formatting_input(current_query, history):
    input_text = ''
    for idx, (query, answer) in history:
        input_text += f"[Round {idx}]\nHuman: {query}\nAssistant: {answer}\n"
    input_text += f"[Round {len(history)}]\nHuman: {current_query}\nAssistant:"
    return input_text
```

### SMILES preprocess

When there involves SMILES notation in your input, we recommend to preprocess the SMILES with the `rdkit` package to canonicalize the SMILES. Here is an example:
```python
from rdkit import Chem
def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=False)
```
or directly:
```python
from rdkit import Chem
def canonicalize_smiles(smiles):
    return Chem.CanonSmiles(smiles, useChiral=True)
```

## Performance

### Chemical Benchmarks

We evaluate the performance of ChemDFM-13B on multiple widely used benchmarks in chemistry. The detailed introduction of the benchmarks can be found in [our paper](https://arxiv.org/abs/2401.14818). The overall performance of ChemDFM-13B is shown below:

![Objective Performance](https://github.com/OpenDFM/ChemDFM/raw/main/docs/static/images/objective_performances.png) 

### Human Evaluation

![Human Evaluation of QA](https://github.com/OpenDFM/ChemDFM/raw/main/docs/static/images/human_evaluation_QA.png)

We mark <font color=#548235>the correct and relevant information</font> in the replies in green, <font color=#C55A11>the correct but irrelevant information</font> in yellow, and <font color=#C00000>the wrong information in red</font>. In addition, **the key points of the answer** are marked in bold if they appear in the reply. 

The results show that while open-sourced LLMs perform well when asked about existing knowledge (Q1), *only ChemDFM can provide correct and comprehensive answers when questions involve new molecules and reactions* (Q2 [\[Yin et al., 2023\]](https://pubs.acs.org/doi/10.1021/jacs.3c07044) & Q3 [\[Dargo et al., 2023\]](https://www.sciencedirect.com/science/article/pii/S1385894723030966))

![Human Evaluation of Dialogue](https://github.com/OpenDFM/ChemDFM/raw/main/docs/static/images/human_evaluation_dialogue.png)

The above conversation is also inspired by [Yin et al.\[2023\]](https://pubs.acs.org/doi/10.1021/jacs.3c07044). During the dialogue, the researcher wants to selectively oxidize one of the two carbonyl groups of a molecule. However, the
initial solution given by ChemDFM results in both carbonyl groups being oxidized. Through the correction given by the researcher, ChemDFM adjusts its proposal and provides two possible solutions. Finally, the researcher chooses to use protecting groups and ChemDFM further details its advice. *In the dialogue, ChemDFM shows promising capabilities regarding error correction (Round 2) and detailing (Round 3) when handling real-world research scenarios.*


**For more examples and analysis, please refer to [our paper](https://arxiv.org/abs/2401.14818).**

## Citation
```bibtex
@article{zhao2025developing,
  title={Developing ChemDFM as a large language foundation model for chemistry},
  author={Zhao, Zihan and Ma, Da and Chen, Lu and Sun, Liangtai and Li, Zihao and Xia, Yi and Chen, Bo and Xu, Hongshen and Zhu, Zichen and Zhu, Su and others},
  journal={Cell Reports Physical Science},
  volume={6},
  number={4},
  year={2025},
  publisher={Elsevier}
}
```

## Disclaimer
Current version of ChemDFM may generate incorrect or misleading information. Please use it with caution and verify the results with domain experts before making any decisions based on the results.

## Contact

If you have any questions or further requests, please contact [Zihan Zhao](mailto:zhao_mengxin@sjtu.edu.cn) and [Lu Chen](mailto:chenlusz@sjtu.edu.cn).
