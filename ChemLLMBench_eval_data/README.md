# How to format these data into inputs

For the jsonline file where the `"few_shots_examples"` key is absent in the data:

```python
text = item['question'] + item['input_format'].format(item['input']) if 'input_format' in item else item['question']
inputs = f"[Round 0]\nHuman: {text.strip()}\nAssistant:"
```

For the jsonline file where the `"few_shots_examples"` key is present in the data:

```python
inputs = f"[Round 0]\nHuman: {raw['question'].strip()}\nAssistant: OK."
for idx, fs in enumerate(raw['few_shots_examples']):
    inputs += f"\n[Round {idx + 1}]\nHuman: {raw['input_format'].format(fs['human'].strip())}\nAssistant: {fs['assistant'].strip()}"
inputs += f"\n[Round {len(raw['few_shots_examples']) + 1}]\nHuman: {raw['input_format'].format(raw['input'].strip())}\nAssistant:"
```