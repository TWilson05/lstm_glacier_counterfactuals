"""Run partial glacier removal inference using saved ensemble checkpoints.

Example: python -m src.partial_glacier_inference --exp_name topographic --member_id 0
"""
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from run_training import EXPERIMENT_CONFIGS
from src.config import DATA_END_YEAR, DATA_START_YEAR, MODELS_DIR, OUTPUT_DATA_DIR
from src.data_utils import align_and_filter, load_raw_csvs, load_scalers, normalize
from src.inference import SEQUENCE_LENGTH
from src.models import EALSTM, StandardLSTM


def run(exp_name, model_type, member_id, batch_size=256):
    features = EXPERIMENT_CONFIGS[exp_name]
    dynamic_cols, static_cols = features['dynamic'], features['static']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = MODELS_DIR / f'{exp_name}_{model_type}_member_{member_id}.pth'
    scalers = load_scalers(MODELS_DIR / f'{exp_name}_scalers.json')
    if scalers.get('static_features', static_cols) != static_cols:
        raise ValueError('Saved scaler static features do not match the experiment configuration')
    model_class = StandardLSTM if model_type == 'lstm' else EALSTM
    model = model_class(len(dynamic_cols), len(static_cols), 256).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    raw_dyn, raw_flow, raw_static = load_raw_csvs(dynamic_cols)
    dyn, _, _, static, stations, dates = align_and_filter(
        raw_dyn, raw_flow, raw_static, static_cols
    )
    glacier_indices = np.flatnonzero(static['glacier_pct'].to_numpy() > 0)
    if not len(glacier_indices):
        raise ValueError('No aligned catchments have positive glacier_pct')
    dyn_array = np.stack([dyn[col].values for col in dynamic_cols], axis=2).astype(np.float32)
    dyn_norm = normalize(dyn_array, scalers['dyn_mean'], scalers['dyn_std'])
    indices = np.arange(len(dates))
    indices = indices[(dates.year >= DATA_START_YEAR) & (dates.year <= DATA_END_YEAR)
                      & (indices >= SEQUENCE_LENGTH)]
    if not len(indices):
        raise ValueError('No dates available after the model lookback period')
    output_dir = OUTPUT_DATA_DIR / 'partial_glacier_removal'
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [dict() for _ in range(11)]
    glacier_col = static_cols.index('glacier_pct')
    print(f'Using {device}; predicting {len(indices)} days for {len(glacier_indices)} glacierized catchments')
    with torch.inference_mode():
        for i in tqdm(glacier_indices, desc='Stations'):
            original = static.iloc[i].values.astype(np.float32)
            stat = np.repeat(original[None, :], 11, axis=0)
            stat[:, glacier_col] = original[glacier_col] * (1 - np.arange(11) / 10)
            stat = torch.tensor(normalize(stat, scalers['stat_mean'], scalers['stat_std']),
                                dtype=torch.float32, device=device)
            predictions = [[] for _ in range(11)]
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                windows = np.array([dyn_norm[t-SEQUENCE_LENGTH:t, i, :] for t in batch])
                dynamic = torch.tensor(windows, dtype=torch.float32, device=device)
                for level in range(11):
                    values = model(dynamic, stat[level].unsqueeze(0).expand(len(batch), -1))
                    predictions[level].extend(values.cpu().numpy().flatten())
            for level in range(11):
                results[level][stations[i]] = predictions[level]
    for level, result in enumerate(results):
        path = output_dir / f'{exp_name}_{model_type}_removal_{level * 10:03d}_member_{member_id}.csv'
        frame = pd.DataFrame(result, index=dates[indices])
        frame.index.name = 'Date'
        frame.to_csv(path)
        print(f'Saved {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp_name', required=True, choices=EXPERIMENT_CONFIGS)
    parser.add_argument('--model_type', default='lstm', choices=['lstm', 'ealstm'])
    parser.add_argument('--member_id', required=True, type=int)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    if args.member_id < 0 or args.batch_size < 1:
        parser.error('member_id must be nonnegative and batch_size must be positive')
    run(args.exp_name, args.model_type, args.member_id, args.batch_size)


if __name__ == '__main__':
    main()
