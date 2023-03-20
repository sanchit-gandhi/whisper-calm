import argparse
import csv
import time

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperProcessor
import evaluate

from modeling_whisper_calm import WhisperForConditionalGeneration

def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate a Whisper model with CALM decoding.")
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="The tolerance delta to within which the early-stopping model's predictions should match the full model's "
             "predictions with probability epsilon.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="The probability epsilon to within which the early-stopping model's predictions should match the full model's "
             "with a tolerance of delta.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tiny.en",
        help="The checkpoint to calibrate. One of `['tiny.en', 'base.en', 'small.en', 'medium.en', 'large-v2']`.",
    )
    parser.add_argument(
        "--max_samples",
        default=4,
        type=int,
        help="Maximum number of samples for the calibration dataset.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        default=4,
        type=int,
        help="Number of workers for the PyTorch dataloader.",
    )
    parser.add_argument(
        "--output_csv_file",
        type=str,
        default="results.csv",
        help="Where to write the benchmark results.",
    )

    args = parser.parse_args()
    return args

def load_and_prepare_data(processor, max_samples=None, dataloader_num_workers=4):
    validation, test = load_dataset("librispeech_asr", "clean", split=["validation", "test"])

    if max_samples is not None:
        # clip the calibration set if necessary
        validation = validation.select(range(max_samples))
        # TODO(SG): remove debugging statement
        test = test.select(range(max_samples))

    def preprocess(batch):
        batch["input_features"] = processor(batch["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_features[0]
        batch["labels"] = processor(text=batch["text"]).input_ids
        return batch

    validation_processed = validation.map(preprocess, remove_columns=validation.column_names)
    test_processed = test.map(preprocess, remove_columns=test.column_names)

    val_dataloader = DataLoader(
        validation_processed.with_format("torch"),
        batch_size=1,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_processed.with_format("torch"),
        batch_size=1,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )
    return val_dataloader, test_dataloader

def main():
    args = parse_args()

    # processors/tokenizers are the same for all models, so just load from tiny and preprocess once
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    val_dataloader, test_dataloader = load_and_prepare_data(
        processor=processor,
        max_samples=args.max_samples,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    # load the model once at the start - we can change the threshold values on the fly
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{args.checkpoint}")
    model.to("cuda").half()

    labels = []
    pred_ids = []

    start = time.time()
    for batch in tqdm(val_dataloader, desc="Initial val step"):
        input_features = batch["input_features"].to("cuda").half()
        generate_out = model.generate(input_features, max_length=128, return_dict_in_generate=True)
        pred_ids.append(generate_out.sequences[0].cpu().numpy())
        labels.append(batch.pop("labels")[0].numpy())

    runtime_static = time.time() - start

    wer = evaluate.load("wer")
    label_str = processor.batch_decode(labels, skip_special_tokens=True, normalize=True)
    pred_str_static = processor.batch_decode(pred_ids, skip_special_tokens=True, normalize=True)
    wer_static = wer.compute(predictions=pred_str_static, references=label_str)

    print(wer_static)

    # specify a grid of exit thresholds in [0, 1]
    lambdas = np.arange(0, 1, 0.05)[::-1]

    # arrays to store our results
    all_runtimes = []
    all_decoder_layers = []

    for lmbda in lambdas:
        model.model.decoder.threshold = lmbda

        pred_ids = []
        decoder_layers = []

        start = time.time()
        for batch in tqdm(val_dataloader):
            input_features = batch["input_features"].to("cuda").half()
            generate_out = model.generate(input_features, max_length=128, return_dict_in_generate=True)
            pred_ids.append(generate_out.sequences[0].cpu().numpy())
            decoder_layers.extend(generate_out.decoder_layers)

        runtime = time.time() - start

        all_runtimes.append(runtime)
        all_decoder_layers.append(np.mean(decoder_layers))

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True, normalize=True)

        distances = []
        for pred, pred_static in zip(pred_str, pred_str_static):
            # compute WER on a sample basis
            distances.append(wer.compute(predictions=pred, references=pred_static))

        # take the expectation
        mean_distance = np.mean(distances)

        num_samples = len(distances)

        # compute p-value through Hoeffding’s inequality
        p_j = np.exp(-2 * num_samples * max(0, args.delta - mean_distance) ** 2)

        # break condition
        if p_j > args.epsilon:
            # compute the WER of our calibrated early-exit model TODO(SG): WER over test set
            wer_calibrated = wer.compute(predictions=pred_str, references=label_str)

            print(f"Static WER: {wer_static:.4f}")
            print(f"Calibrated WER: {wer_calibrated:.4f}")

            print(f"Static Runtime: {runtime_static:.4f}")
            print(f"Calibrated Runtime: {runtime:.4f}")
            break

    # Save the results
    headers = ["Threshold", "Runtime", "Avg Decoder Layers", ""]
    with open(args.output_csv_file, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        # write the headers
        writer.writerow(headers)
        # write the data
        for i in range(len(all_runtimes)):
            writer.writerow([lambdas[i], all_runtimes[i], all_decoder_layers[i]])

if __name__ == "__main__":
    main()
