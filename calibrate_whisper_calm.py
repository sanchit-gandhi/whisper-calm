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
        default="medium.en",
        help="The checkpoint to calibrate. One of `['tiny.en', 'base.en', 'small.en', 'medium.en', 'large-v2']`.",
    )
    parser.add_argument(
        "--max_samples",
        default=100,
        type=int,
        help="Maximum number of samples for the calibration dataset.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        default=4,
        type=int,
        help="Number of workers for the PyTorch dataloader.",
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

def load_and_prepare_data(processor, max_samples=None, preprocessing_num_workers=4, dataloader_num_workers=4):
    validation = load_dataset("librispeech_asr", "clean", split="validation")

    if max_samples is not None:
        # clip the calibration set if necessary
        validation = validation.select(range(max_samples))

    def preprocess(batch):
        batch["input_features"] = processor(batch["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_features[0]
        batch["labels"] = processor(text=batch["text"]).input_ids
        return batch

    validation_processed = validation.map(preprocess, remove_columns=validation.column_names, num_proc=preprocessing_num_workers)

    val_dataloader = DataLoader(
        validation_processed.with_format("torch"),
        batch_size=1,
        num_workers=dataloader_num_workers,
        pin_memory=True,
    )
    return val_dataloader

def main():
    args = parse_args()

    # processors/tokenizers are the same for all models, so just load from tiny and preprocess once
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    val_dataloader = load_and_prepare_data(
        processor=processor,
        max_samples=args.max_samples,
        preprocessing_num_workers=args.preprocessing_num_workers,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    # load the model once at the start - we can change the threshold values on the fly
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{args.checkpoint}")
    model.to("cuda").half().eval()

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

    # arrays to store our results
    all_runtimes = [runtime_static]
    all_decoder_layers = [model.config.decoder_layers]
    all_distances = [0.0]
    all_wers = [wer_static]
    all_pjs = [1.0]

    # specify a grid of exit thresholds in [0, 1] -> in practice anything < 0.95 gives practically random outputs so clip here
    lambdas = np.arange(0.925, 1.01, 0.0125)[::-1]

    for lmbda in lambdas[1:]:
        model.model.decoder.threshold = lmbda

        pred_ids = []
        decoder_layers = []

        start = time.time()
        for batch in tqdm(val_dataloader, desc=f"Threshold = {lmbda:.3f}"):
            input_features = batch["input_features"].to("cuda").half()
            generate_out = model.generate(input_features, max_length=128, return_dict_in_generate=True)
            pred_ids.append(generate_out.sequences[0].cpu().numpy())
            decoder_layers.extend(generate_out.decoder_layers)

        runtime = time.time() - start

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True, normalize=True)

        distances = []
        for pred, pred_static in zip(pred_str, pred_str_static):
            # compute WER on a sample basis
            distances.append(wer.compute(predictions=[pred], references=[pred_static]))

        # take the expectation
        mean_distance = np.mean(distances)
        num_samples = len(distances)

        # compute p-value through Hoeffding’s inequality
        p_j = np.exp(-2 * num_samples * max(0, args.delta - mean_distance) ** 2)

        # exit condition - skip for whisper since it breaks
        # if p_j > args.epsilon

        # compute the WER of our calibrated early-exit model TODO(SG): WER over test set
        wer_calibrated = wer.compute(predictions=pred_str, references=label_str)

        all_runtimes.append(runtime)
        all_decoder_layers.append(np.mean(decoder_layers))
        all_distances.append(mean_distance)
        all_wers.append(wer_calibrated)
        all_pjs.append(p_j)

    # Save the results
    headers = ["Threshold", "Runtime", "Avg Decoder Layers", "Exp Distance", "WER", "p_j"]
    with open(args.output_csv_file, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        # write the headers
        writer.writerow(headers)
        # write the data
        for i in range(len(all_runtimes)):
            writer.writerow([round(lambdas[i], 3), round(all_runtimes[i], 1), round(all_decoder_layers[i], 1), round(all_distances[i], 2), 100 * round(all_wers[i], 4), round(all_pjs[i], 4)])

if __name__ == "__main__":
    main()
