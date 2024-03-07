import torch
import logging
import argparse
from transformers import LlamaTokenizer, LlamaForCausalLM

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the dialogue capability of ChemDFM-13B through CLI.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to sample"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help='Specify temperature',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help='Specify num of top k',
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help='Specify num of top p',
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beam groups',
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help='Specify num of repetition penalty',
    )
    parser.add_argument(
        "--max_round",
        type=int,
        default=0
    )

    args = parser.parse_args()
    return args


def response(model, tokenizer, inputs, args, device):
    tokenized_inputs = tokenizer(inputs, return_tensors="pt").to(device).input_ids
    generate_ids = model.generate(tokenized_inputs,
                                  max_new_tokens=args.max_new_tokens,
                                  do_sample=args.do_sample,
                                  temperature=args.temperature,
                                  top_k=args.top_k,
                                  top_p=args.top_p,
                                  num_beams=args.num_beams,
                                  num_beam_groups=args.num_beam_groups,
                                  repetition_penalty=args.repetition_penalty,
                                  num_return_sequences=1,
                                  eos_token_id=tokenizer.eos_token_id)
    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    returns = result[0][len(inputs):]
    return returns.strip()


def encapsulate_history_and_current_input_into_prompt(history, cur_input):
    prompt = ""
    for idx, (human, assistant) in enumerate(history):
        prompt += f"[Round {idx}]\nHuman: {human}\nAssistant: {assistant}\n"
    prompt += f"[Round {len(history)}]\nHuman: {cur_input}\nAssistant:"
    return prompt


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, device_map='auto', torch_dtype=torch.float16)
    model.eval()

    history = []
    while True:
        inputs = input("Query: ")

        if inputs.strip() == '/clear':
            print()
            print('------------------- new conversation start -------------------')
            print()
            history = []
        elif inputs.strip() == '/back':
            history = history[:-1]
        else:
            prompt = encapsulate_history_and_current_input_into_prompt(history[-args.max_round:], inputs)
            returns = response(model, tokenizer, prompt, args, device)

            if args.max_round > 0:
                history.append((inputs, returns))


if __name__ == "__main__":
    main()
