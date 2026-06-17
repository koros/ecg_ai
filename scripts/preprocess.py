#!/usr/bin/env python

"""
Convert LTAFDB records into fixed-length ECG windows.

Input:
    data/raw/ltafdb/

Output:
    data/processed/
        X.npy
        y.npy

Each sample:
    X.shape = (n_windows, window_samples, n_channels)
    y.shape = (n_windows,)
"""

from pathlib import Path
import argparse

import numpy as np
import wfdb
from tqdm import tqdm

FS = 128  # Hz


def build_label_array(n_samples, ann):

    labels = np.zeros(
        n_samples,
        dtype=np.uint8,
    )

    rhythm_events = []

    for sample, note in zip(
        ann.sample,
        ann.aux_note,
    ):

        note = note.strip()

        if note:
            rhythm_events.append(
                (sample, note)
            )

    for idx in range(
        len(rhythm_events) - 1
    ):

        start, rhythm = rhythm_events[idx]
        end, _ = rhythm_events[idx + 1]

        if "(AFIB" in rhythm:
            labels[start:end] = 1

    if rhythm_events:

        start, rhythm = rhythm_events[-1]

        if "(AFIB" in rhythm:
            labels[start:] = 1

    return labels

def extract_windows(
    signal,
    labels,
    window_seconds=10,
    stride_seconds=10,
):
    window_size = window_seconds * FS
    stride = stride_seconds * FS

    X = []
    y = []

    for start in range(
        0,
        len(signal) - window_size,
        stride,
    ):
        end = start + window_size

        window = signal[start:end]
        window_label = labels[start:end]

        # Majority vote
        label = int(window_label.mean() > 0.5)

        X.append(window.astype(np.float32))
        y.append(label)

    return X, y


def process_record(
    record_path,
    window_seconds,
    stride_seconds,
):
    record = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), "atr")

    signal = record.p_signal

    labels = build_label_array(
        signal.shape[0],
        ann,
    )

    return extract_windows(
        signal,
        labels,
        window_seconds,
        stride_seconds,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--data-dir", default="data/raw/ltafdb", )
    parser.add_argument( "--output-dir", default="data/processed", )
    parser.add_argument( "--window-seconds", type=int, default=10, )
    parser.add_argument( "--stride-seconds", type=int, default=10, ) 
    args = parser.parse_args()
    data_dir = Path(args.data_dir) 
    output_dir = Path(args.output_dir) 
    output_dir.mkdir( parents=True, exist_ok=True, )
    records = sorted(data_dir.glob("*.hea")) 
    print(f"Found {len(records)} records")
    total_windows = 0
    for header_file in tqdm(records):
        record_base = header_file.with_suffix("")

        X, y = process_record(
            record_base,
            args.window_seconds,
            args.stride_seconds,
        )   

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        y = np.asarray(
            y,
            dtype=np.uint8,
            )

        output_file = (
            output_dir /
            f"{record_base.name}.npz"
            )
        np.savez_compressed(
            output_file,
            X=X,
            y=y,
            )   

        total_windows += len(y)

        print(
            f"{record_base.name}: "
            f"{len(y)} windows"
            )

        print(
        f"\nSaved {len(records)} files"
        )

        print(
        f"Total windows: {total_windows}"
        )

        print(
        f"Output directory: {output_dir}"
        ) 

   
if __name__ == "__main__":
    main()
