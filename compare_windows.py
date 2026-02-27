#!/usr/bin/env python3
"""Compare npz outputs between window=10 and window=30"""
import numpy as np

def compare_scheme_matrices():
    d10 = np.load('alfafa/matrix/scheme_matrices.npz', allow_pickle=True)
    d30 = np.load('alfafa/matrix_3/scheme_matrices.npz', allow_pickle=True)
    users = list(d10['users'])

    print('=' * 70)
    print('SCHEME MATRICES: window=10 vs window=3')
    print('=' * 70)

    rankings = []

    for key in sorted(d10.files):
        if key == 'users':
            continue
        m10 = d10[key]
        m30 = d30[key]

        if np.allclose(m10, m30, atol=1e-10):
            print(f'  {key}: IDENTICAL')
            continue

        diff = m30 - m10
        abs_diff = np.abs(diff)
        valid = ~(np.isnan(m10) | np.isnan(m30))
        if not valid.any():
            print(f'  {key}: all NaN')
            continue

        abs_valid = abs_diff[valid]
        total = abs_valid.sum()
        rankings.append((total, key))

        print(f'\n  {key}: mean|delta|={abs_valid.mean():.4f}, max|delta|={abs_valid.max():.4f}, total|delta|={total:.2f}')

        # Top 3 biggest changes
        temp = abs_diff.copy()
        temp[~valid] = 0
        for rank in range(3):
            idx = np.unravel_index(temp.argmax(), temp.shape)
            i, j = idx
            if temp[i, j] == 0:
                break
            print(f'    #{rank+1} {users[i]} -> {users[j]}: {m10[i,j]:.3f} -> {m30[i,j]:.3f} (delta={diff[i,j]:+.3f})')
            temp[i, j] = 0

    print('\n' + '-' * 70)
    print('RANKING by total absolute change:')
    rankings.sort(reverse=True)
    for total, key in rankings:
        print(f'  {key}: {total:.2f}')


def compare_time_series():
    t10 = np.load('alfafa/matrix/time_series_tensors.npz', allow_pickle=True)
    t30 = np.load('alfafa/matrix_3/time_series_tensors.npz', allow_pickle=True)
    users = list(t10['users'])
    labels = list(t10['labels'])

    print('\n' + '=' * 70)
    print('TIME-SERIES TENSORS: window=10 vs window=3')
    print('=' * 70)

    rankings = []

    for key in sorted(t10.files):
        if key in ('users', 'labels'):
            continue
        ten10 = t10[key]
        ten30 = t30[key]

        if np.allclose(ten10, ten30, atol=1e-10):
            print(f'  {key}: IDENTICAL')
            continue

        diff = ten30 - ten10
        abs_diff = np.abs(diff)
        total = abs_diff.sum()
        rankings.append((total, key))

        seg_totals = [abs_diff[s].sum() for s in range(ten10.shape[0])]
        top_seg = np.argmax(seg_totals)

        print(f'\n  {key}: total|delta|={total:.2f}, biggest segment: {labels[top_seg]} ({seg_totals[top_seg]:.2f})')

        # Top 3 cell changes across all segments
        temp = abs_diff.copy()
        for rank in range(3):
            idx = np.unravel_index(temp.argmax(), temp.shape)
            s, i, j = idx
            if temp[s, i, j] == 0:
                break
            print(f'    #{rank+1} [{labels[s]}] {users[i]} -> {users[j]}: {ten10[s,i,j]:.3f} -> {ten30[s,i,j]:.3f} (delta={diff[s,i,j]:+.3f})')
            temp[s, i, j] = 0

    print('\n' + '-' * 70)
    print('RANKING by total absolute change:')
    rankings.sort(reverse=True)
    for total, key in rankings:
        print(f'  {key}: {total:.2f}')

    # Identify which panorama PNGs are most worth looking at
    print('\n' + '=' * 70)
    print('RECOMMENDED PNGs TO COMPARE (most change)')
    print('=' * 70)

    # Combine scheme + time-series rankings
    all_rankings = rankings[:]  # time-series already here

    # Map tensor keys to PNG filenames
    key_to_files = {}
    for total, key in all_rankings:
        scheme = key.replace('s', '').replace('_', '-')
        key_to_files[key] = (
            f'time_series_{scheme}.png',
            f'derivative_{scheme}.png'
        )

    print('\nTime-series panoramas with most change (compare matrix/ vs matrix_30/):')
    for total, key in sorted(all_rankings, reverse=True)[:8]:
        ts_file, deriv_file = key_to_files[key]
        print(f'  {ts_file}  (total|delta|={total:.2f})')

    print('\nDerivative panoramas to re-check:')
    for total, key in sorted(all_rankings, reverse=True)[:8]:
        ts_file, deriv_file = key_to_files[key]
        print(f'  {deriv_file}')


if __name__ == '__main__':
    compare_scheme_matrices()
    compare_time_series()
