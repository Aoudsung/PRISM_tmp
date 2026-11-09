# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
from pathlib import Path

from datasets import Dataset

from utils import make_prefix


DEFAULT_TRAIN_SIZE = 1000
DEFAULT_VAL_SIZE = 300

num2label = {
    1: "A",
    2: "A, B",
    3: "A, B, C",
    4: "A, B, C, D",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CircuitDecoding JSONL data to parquet.")
    parser.add_argument("--local_dir", required=True, help="Workspace root containing the raw dataset directory.")
    parser.add_argument("--dataset", type=str, default="CircuitDecoding")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cand", type=int, default=20)
    parser.add_argument("--input_file", type=str, default=None, help="Optional path to the raw JSONL file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory for parquet files.")
    parser.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--train_output", type=str, default=None)
    parser.add_argument("--val_output", type=str, default=None)

    args = parser.parse_args()

    dataset_root = Path(args.local_dir).expanduser() / args.dataset
    input_path = (
        Path(args.input_file).expanduser()
        if args.input_file is not None
        else dataset_root / f"cd_raw_2_circuits_{args.cand}_cand.jsonl"
    )
    output_dir = Path(args.output_dir).expanduser() if args.output_dir is not None else dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)

    org_dataset = load_jsonl(input_path)
    if len(org_dataset) < args.train_size + args.val_size:
        raise ValueError(
            f"Not enough samples in {input_path}: requested train_size={args.train_size} and "
            f"val_size={args.val_size}, but only found {len(org_dataset)} records."
        )

    train_data = org_dataset[: args.train_size]
    val_data = org_dataset[-args.val_size :]

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, dataset=args.dataset, belief_cot=False)
            controller = example["controller"]
            question = question.format(
                num_circuits=controller["num_circuits"],
                circuit_labels=num2label[controller["num_circuits"]],
                num_candidates=len(controller["candidates"]),
                num_inputs=controller["num_inputs"],
                candidate_list_str="\n".join(
                    [f"{i + 1}. {item[2:-2].strip()}" for i, item in enumerate(controller["candidates"])]
                ),
            )
            solution = {
                "target": controller["circuit_joint_lookup"],
            }

            data = {
                "answer": {
                    "target": controller["circuit_joint_lookup"],
                    "num_circuits": controller["num_circuits"],
                    "num_inputs": controller["num_inputs"],
                    "candidates": controller["candidates"],
                },
                "data_source": "cd",
                "prompt": [{"role": "user", "content": question}],
                "ability": "fact-reasoning",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data

        return process_fn

    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(function=make_map_fn(args.split), with_indices=True)
    train_output = Path(args.train_output).expanduser() if args.train_output is not None else output_dir / f"train__cand_{args.cand}.parquet"
    train_dataset.to_parquet(train_output)
    print(f"Saved train dataset to {train_output} ({len(train_dataset)} samples)")

    val_dataset = Dataset.from_list(val_data)
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)
    val_output = Path(args.val_output).expanduser() if args.val_output is not None else output_dir / f"val__cand_{args.cand}.parquet"
    val_dataset.to_parquet(val_output)
    print(f"Saved val dataset to {val_output} ({len(val_dataset)} samples)")
