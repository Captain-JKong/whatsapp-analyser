#!/usr/bin/env python3
"""
Adjacency Matrix Analysis - Multiple algorithms for computing user interaction matrices
Different schemes for understanding network dynamics in group chats
"""

import re
import sys
import os
import argparse
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class AdjacencyMatrixAnalyzer:
    def __init__(self, input_file, exclude_bottom_n=3, window_size=10):
        self.input_file = input_file
        self.messages = []
        self.users = set()
        self.exclude_bottom_n = exclude_bottom_n
        self.window_size = window_size
        self.excluded_users = set()
        self.parse_messages()
        if self.exclude_bottom_n > 0:
            self.filter_bottom_users()
    
    def parse_messages(self):
        """Parse messages from WhatsApp export file"""
        pattern = r'\[(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}:\d{2}(?:\s+(?:am|pm|AM|PM))?)\] ([^:]+): (.*)'
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        current_message = None
        
        for line in lines:
            match = re.match(pattern, line)
            if match:
                if current_message:
                    self.messages.append(current_message)
                
                date_str, time_str, sender, text = match.groups()
                time_str = time_str.strip()
                
                try:
                    if any(x in time_str.lower() for x in ['am', 'pm']):
                        timestamp = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %I:%M:%S %p")
                    else:
                        timestamp = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M:%S")
                except ValueError:
                    continue
                
                current_message = {
                    'timestamp': timestamp,
                    'sender': sender,
                    'text': text
                }
                self.users.add(sender)
            else:
                if current_message and line.strip():
                    current_message['text'] += '\n' + line
        
        if current_message:
            self.messages.append(current_message)
    
    def filter_bottom_users(self):
        """Filter out N users with the lowest message counts"""
        user_message_counts = Counter(msg['sender'] for msg in self.messages)
        
        if len(user_message_counts) <= self.exclude_bottom_n:
            print(f"⚠️  Warning: Requested to exclude {self.exclude_bottom_n} users, but only {len(user_message_counts)} users exist.")
            return
        
        bottom_users = [user for user, count in user_message_counts.most_common()[::-1][:self.exclude_bottom_n]]
        self.excluded_users = set(bottom_users)
        
        original_count = len(self.messages)
        self.messages = [msg for msg in self.messages if msg['sender'] not in self.excluded_users]
        self.users = set(msg['sender'] for msg in self.messages)
        
        print(f"\n🚫 Excluded {self.exclude_bottom_n} users with lowest message counts:")
        for user in bottom_users:
            count = user_message_counts[user]
            print(f"   - {user}: {count} messages")
        print(f"   Total messages excluded: {original_count - len(self.messages)}\n")
    
    # ========== SCHEME 01: Counter-based (immediate replies only) ==========
    
    def compute_01_counter_based(self):
        """01a: Absolute immediate reply counts"""
        all_users = sorted(self.users)
        user_to_idx = {user: idx for idx, user in enumerate(all_users)}
        matrix = np.zeros((len(all_users), len(all_users)))
        
        for i in range(len(self.messages) - 1):
            sender = self.messages[i]['sender']
            next_sender = self.messages[i + 1]['sender']
            
            if sender != next_sender:
                matrix[user_to_idx[sender], user_to_idx[next_sender]] += 1
        
        return matrix, all_users
    
    def normalize_receiver(self, matrix, all_users):
        """01b: Normalize by receiver's total messages"""
        user_message_counts = Counter(msg['sender'] for msg in self.messages)
        matrix_norm = matrix.copy()
        
        for j, user in enumerate(all_users):
            total = user_message_counts.get(user, 1)
            if total > 0:
                matrix_norm[:, j] /= total
        
        return matrix_norm
    
    def normalize_sender(self, matrix, all_users):
        """01c: Normalize by sender's total messages"""
        user_message_counts = Counter(msg['sender'] for msg in self.messages)
        matrix_norm = matrix.copy()
        
        for i, user in enumerate(all_users):
            total = user_message_counts.get(user, 1)
            if total > 0:
                matrix_norm[i, :] /= total
        
        return matrix_norm
    
    def normalize_double(self, matrix, all_users):
        """01d: Normalize by sender then column"""
        matrix_norm = self.normalize_sender(matrix, all_users)
        
        column_sums = matrix_norm.sum(axis=0)
        column_sums[column_sums == 0] = 1
        matrix_norm = matrix_norm / column_sums[np.newaxis, :]
        
        return matrix_norm
    
    # ========== SCHEME 02: Time-decay immediate replies ==========
    
    def compute_02_time_decay_immediate(self):
        """02: Immediate reply with time decay 0.5-0.5*tanh(x/4-4)"""
        all_users = sorted(self.users)
        user_to_idx = {user: idx for idx, user in enumerate(all_users)}
        matrix = np.zeros((len(all_users), len(all_users)))
        
        for i in range(len(self.messages) - 1):
            sender = self.messages[i]['sender']
            next_sender = self.messages[i + 1]['sender']
            
            if sender != next_sender:
                time_diff = (self.messages[i + 1]['timestamp'] - self.messages[i]['timestamp']).total_seconds() / 60  # minutes
                score = 0.5 - 0.5 * np.tanh(time_diff / 4 - 4)
                matrix[user_to_idx[sender], user_to_idx[next_sender]] += score
        
        return matrix, all_users
    
    # ========== SCHEME 03: Time-decay with 10-message window ==========
    
    def compute_03_time_decay_window(self):
        """03: Non-immediate reply within N most recent messages"""
        all_users = sorted(self.users)
        user_to_idx = {user: idx for idx, user in enumerate(all_users)}
        matrix = np.zeros((len(all_users), len(all_users)))
        
        for i in range(len(self.messages)):
            sender = self.messages[i]['sender']
            
            # Look at previous N messages
            start_idx = max(0, i - self.window_size)
            for j in range(start_idx, i):
                prev_sender = self.messages[j]['sender']
                
                if prev_sender != sender:
                    time_diff = (self.messages[i]['timestamp'] - self.messages[j]['timestamp']).total_seconds() / 60
                    score = 0.5 - 0.5 * np.tanh(time_diff / 4 - 4)
                    matrix[user_to_idx[prev_sender], user_to_idx[sender]] += score
        
        return matrix, all_users
    
    # ========== SCHEME 04: Late-night + DM multiplier (on top of 03) ==========
    
    def compute_04_late_night(self):
        """04: Scheme 03 with 1.2x late-night (11pm-4am) + 2x DM multiplier"""
        all_users = sorted(self.users)
        user_to_idx = {user: idx for idx, user in enumerate(all_users)}
        matrix = np.zeros((len(all_users), len(all_users)))
        
        dm_chunk_count = 0
        dm_pairs = Counter()
        # Track which start_idx we've already seen to find distinct DM conversations
        dm_seen_starts = set()
        dm_convos = []  # list of (pair, first_msg)
        
        for i in range(len(self.messages)):
            sender = self.messages[i]['sender']
            hour = self.messages[i]['timestamp'].hour
            
            # Late-night multiplier: 11pm (23) to 4am (3)
            night_mult = 1.2 if (hour >= 23 or hour < 4) else 1.0
            
            # Look at previous N messages (same as scheme 03)
            start_idx = max(0, i - self.window_size)
            window = self.messages[start_idx:i + 1]
            
            # DM multiplier: if the window only has 2 unique senders
            unique_senders = set(msg['sender'] for msg in window)
            dm_mult = 1.0
            if len(unique_senders) == 2 and len(window) >= self.window_size:
                dm_mult = 2.0
                dm_chunk_count += 1
                pair = tuple(sorted(unique_senders))
                dm_pairs[pair] += 1
                # Record the first message of a new DM stretch (when start_idx first enters DM)
                if start_idx not in dm_seen_starts:
                    dm_seen_starts.add(start_idx)
                    first = self.messages[start_idx]
                    dm_convos.append((pair, first))
            
            for j in range(start_idx, i):
                prev_sender = self.messages[j]['sender']
                
                if prev_sender != sender:
                    time_diff = (self.messages[i]['timestamp'] - self.messages[j]['timestamp']).total_seconds() / 60
                    score = 0.5 - 0.5 * np.tanh(time_diff / 4 - 4)
                    matrix[user_to_idx[prev_sender], user_to_idx[sender]] += score * night_mult * dm_mult
        
        print(f"    [04 debug] DM-multiplier (2x) applied to {dm_chunk_count} message windows")
        for pair, count in dm_pairs.most_common():
            print(f"      {pair[0]} & {pair[1]}: {count} windows")
        if dm_convos:
            print(f"    [04 debug] DM conversation start messages ({len(dm_convos)} distinct):")
            for pair, msg in dm_convos:
                ts = msg['timestamp'].strftime('%Y-%m-%d %H:%M')
                text = msg['text'][:80]
                print(f"      [{ts}] {msg['sender']}: {text}")
        
        return matrix, all_users
    
    # ========== SCHEME 05: Exclusivity index ==========
    
    def compute_05_exclusivity(self):
        """05: Ratio of A→B interactions vs A→everyone. High = tunnel vision on B."""
        # First compute raw interaction counts (scheme 03 style)
        all_users = sorted(self.users)
        user_to_idx = {user: idx for idx, user in enumerate(all_users)}
        raw_matrix = np.zeros((len(all_users), len(all_users)))
        
        for i in range(len(self.messages)):
            sender = self.messages[i]['sender']
            start_idx = max(0, i - self.window_size)
            for j in range(start_idx, i):
                prev_sender = self.messages[j]['sender']
                if prev_sender != sender:
                    time_diff = (self.messages[i]['timestamp'] - self.messages[j]['timestamp']).total_seconds() / 60
                    score = 0.5 - 0.5 * np.tanh(time_diff / 4 - 4)
                    # sender is paying attention to prev_sender (sender replied after prev_sender)
                    raw_matrix[user_to_idx[sender], user_to_idx[prev_sender]] += score
        
        # Compute exclusivity: ratio of A→B / sum(A→everyone)
        matrix = np.zeros_like(raw_matrix)
        for i in range(len(all_users)):
            row_sum = raw_matrix[i, :].sum()
            if row_sum > 0:
                matrix[i, :] = raw_matrix[i, :] / row_sum
        
        return matrix, all_users
    
    # ========== SCHEME 06: First-response latency consistency ==========
    
    def compute_06_response_consistency(self):
        """06: Three matrices - mean reply time, std-dev, and priority score.
        Priority score: 1 / (1 + sqrt(mean * std)) - high when BOTH mean and std are small.
        Uses geometric mean under the square root to balance the two metrics."""
        all_users = sorted(self.users)
        user_to_idx = {user: idx for idx, user in enumerate(all_users)}
        n = len(all_users)
        
        # Collect reply times for each directed pair
        reply_times = [[[] for _ in range(n)] for _ in range(n)]
        
        for i in range(len(self.messages)):
            sender = self.messages[i]['sender']
            start_idx = max(0, i - self.window_size)
            for j in range(start_idx, i):
                prev_sender = self.messages[j]['sender']
                if prev_sender != sender:
                    time_diff = (self.messages[i]['timestamp'] - self.messages[j]['timestamp']).total_seconds() / 60
                    reply_times[user_to_idx[prev_sender]][user_to_idx[sender]].append(time_diff)
        
        matrix_mean = np.full((n, n), np.nan)
        matrix_std = np.full((n, n), np.nan)
        matrix_priority = np.full((n, n), np.nan)
        
        for i in range(n):
            for j in range(n):
                times = reply_times[i][j]
                if len(times) >= 3:
                    mean_t = np.mean(times)
                    std_t = np.std(times)
                    matrix_mean[i][j] = mean_t
                    matrix_std[i][j] = std_t
                    # Priority: high score when both mean and std are small
                    # sqrt(mean * std) is the geometric mean, gives balanced weighting
                    matrix_priority[i][j] = 1.0 / (1.0 + np.sqrt(mean_t * std_t))
        
        return matrix_mean, matrix_std, matrix_priority, all_users
    
    # ========== SCHEME 07: Asymmetric reciprocity ("Chase" metric) ==========
    
    def compute_07_asymmetry(self):
        """07: Matrix[A,B] - Matrix[B,A] from scheme 04 — positive = A chases B."""
        matrix_04, all_users = self.compute_04_late_night()
        
        # Normalize by sender first (same as 'd' normalization) so volumes are comparable
        matrix_04_norm = self.normalize_double(matrix_04, all_users)
        
        # Asymmetry: M[A,B] - M[B,A] (positive means A→B > B→A)
        n = len(all_users)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i, j] = matrix_04_norm[i, j] - matrix_04_norm[j, i]
        
        return matrix, all_users
    
    def create_heatmap(self, matrix, all_users, title, filename, output_dir, cmap='YlOrRd', diverging=False):
        """Create and save heatmap visualization"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(max(12, len(all_users) * 0.35), max(10, len(all_users) * 0.35)))
        
        # Handle NaN: render as light gray
        current_cmap = plt.get_cmap(cmap).copy()
        current_cmap.set_bad(color='#e0e0e0')
        
        if diverging:
            matrix_display = matrix.copy()
            abs_max = max(abs(np.nanmin(matrix_display)), abs(np.nanmax(matrix_display)))
            vmin, vmax = -abs_max, abs_max
            cbar_label = 'Score'
        else:
            # Use log scale for better visibility (NaN stays NaN through log1p)
            matrix_display = np.log1p(matrix)
            vmin, vmax = None, None
            cbar_label = 'Adjacency Score (log scale)'
        
        im = ax.imshow(matrix_display, cmap=current_cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        ax.set_xticks(range(len(all_users)))
        ax.set_yticks(range(len(all_users)))
        ax.set_xticklabels(all_users, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(all_users, fontsize=9)
        
        ax.set_xlabel('To (Receiver)', fontsize=11, fontweight='bold')
        ax.set_ylabel('From (Sender)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        
        cbar = plt.colorbar(im, ax=ax, label=cbar_label)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_all_schemes(self, output_dir='matrix'):
        """Generate all 16 adjacency matrix variants"""
        print("📊 Computing adjacency matrices...\n")
        os.makedirs(output_dir, exist_ok=True)
        
        # Scheme 01: Counter-based
        print("  Computing 01abcd (counter-based)...")
        matrix_01, users = self.compute_01_counter_based()
        self.create_heatmap(matrix_01, users, 'Scheme 01a: Immediate Replies (Absolute)', '01a', output_dir)
        self.create_heatmap(self.normalize_receiver(matrix_01, users), users, 'Scheme 01b: Immediate Replies (Receiver-Normalized)', '01b', output_dir)
        self.create_heatmap(self.normalize_sender(matrix_01, users), users, 'Scheme 01c: Immediate Replies (Sender-Normalized)', '01c', output_dir)
        self.create_heatmap(self.normalize_double(matrix_01, users), users, 'Scheme 01d: Immediate Replies (Double-Normalized)', '01d', output_dir)
        
        # Scheme 02: Time-decay immediate
        print("  Computing 02abcd (time-decay immediate)...")
        matrix_02, _ = self.compute_02_time_decay_immediate()
        self.create_heatmap(matrix_02, users, 'Scheme 02a: Time-Decay Immediate (Absolute)', '02a', output_dir)
        matrix_02_norm_r = self.normalize_receiver(matrix_02, users)
        self.create_heatmap(matrix_02_norm_r, users, 'Scheme 02b: Time-Decay Immediate (Receiver-Norm)', '02b', output_dir)
        matrix_02_norm_s = self.normalize_sender(matrix_02, users)
        self.create_heatmap(matrix_02_norm_s, users, 'Scheme 02c: Time-Decay Immediate (Sender-Norm)', '02c', output_dir)
        matrix_02_double = np.zeros_like(matrix_02_norm_s)
        if matrix_02_norm_s.sum() > 0:
            col_sums = matrix_02_norm_s.sum(axis=0)
            col_sums[col_sums == 0] = 1
            matrix_02_double = matrix_02_norm_s / col_sums[np.newaxis, :]
        self.create_heatmap(matrix_02_double, users, 'Scheme 02d: Time-Decay Immediate (Double-Norm)', '02d', output_dir)
        
        # Scheme 03: Time-decay window
        print("  Computing 03abcd (time-decay 10-message window)...")
        matrix_03, _ = self.compute_03_time_decay_window()
        self.create_heatmap(matrix_03, users, 'Scheme 03a: Time-Decay 10-Window (Absolute)', '03a', output_dir)
        matrix_03_norm_r = self.normalize_receiver(matrix_03, users)
        self.create_heatmap(matrix_03_norm_r, users, 'Scheme 03b: Time-Decay 10-Window (Receiver-Norm)', '03b', output_dir)
        matrix_03_norm_s = self.normalize_sender(matrix_03, users)
        self.create_heatmap(matrix_03_norm_s, users, 'Scheme 03c: Time-Decay 10-Window (Sender-Norm)', '03c', output_dir)
        matrix_03_double = np.zeros_like(matrix_03_norm_s)
        if matrix_03_norm_s.sum() > 0:
            col_sums = matrix_03_norm_s.sum(axis=0)
            col_sums[col_sums == 0] = 1
            matrix_03_double = matrix_03_norm_s / col_sums[np.newaxis, :]
        self.create_heatmap(matrix_03_double, users, 'Scheme 03d: Time-Decay 10-Window (Double-Norm)', '03d', output_dir)
        
        # Scheme 04: Late-night + DM multiplier (with abcd)
        print("  Computing 04abcd (late-night + DM multiplier)...")
        matrix_04, _ = self.compute_04_late_night()
        self.create_heatmap(matrix_04, users, 'Scheme 04a: Night 1.2x + DM 2x (Absolute)', '04a', output_dir)
        matrix_04_norm_r = self.normalize_receiver(matrix_04, users)
        self.create_heatmap(matrix_04_norm_r, users, 'Scheme 04b: Night 1.2x + DM 2x (Receiver-Norm)', '04b', output_dir)
        matrix_04_norm_s = self.normalize_sender(matrix_04, users)
        self.create_heatmap(matrix_04_norm_s, users, 'Scheme 04c: Night 1.2x + DM 2x (Sender-Norm)', '04c', output_dir)
        matrix_04_double = np.zeros_like(matrix_04_norm_s)
        if matrix_04_norm_s.sum() > 0:
            col_sums = matrix_04_norm_s.sum(axis=0)
            col_sums[col_sums == 0] = 1
            matrix_04_double = matrix_04_norm_s / col_sums[np.newaxis, :]
        self.create_heatmap(matrix_04_double, users, 'Scheme 04d: Night 1.2x + DM 2x (Double-Norm)', '04d', output_dir)
        
        # Scheme 05: Exclusivity index (with abcd)
        print("  Computing 05abcd (exclusivity index)...")
        matrix_05, _ = self.compute_05_exclusivity()
        self.create_heatmap(matrix_05, users, 'Scheme 05a: Exclusivity Index (Absolute)', '05a', output_dir)
        matrix_05_norm_r = self.normalize_receiver(matrix_05, users)
        self.create_heatmap(matrix_05_norm_r, users, 'Scheme 05b: Exclusivity Index (Receiver-Norm)', '05b', output_dir)
        matrix_05_norm_s = self.normalize_sender(matrix_05, users)
        self.create_heatmap(matrix_05_norm_s, users, 'Scheme 05c: Exclusivity Index (Sender-Norm)', '05c', output_dir)
        matrix_05_double = np.zeros_like(matrix_05_norm_s)
        if matrix_05_norm_s.sum() > 0:
            col_sums = matrix_05_norm_s.sum(axis=0)
            col_sums[col_sums == 0] = 1
            matrix_05_double = matrix_05_norm_s / col_sums[np.newaxis, :]
        self.create_heatmap(matrix_05_double, users, 'Scheme 05d: Exclusivity Index (Double-Norm)', '05d', output_dir)
        
        # Scheme 06: Response latency (mean, std, priority)
        print("  Computing 06abc (response latency)...")
        matrix_06_mean, matrix_06_std, matrix_06_priority, _ = self.compute_06_response_consistency()
        self.create_heatmap(matrix_06_mean, users, 'Scheme 06a: Mean Reply Time (min)', '06a', output_dir, cmap='YlOrRd_r')
        self.create_heatmap(matrix_06_std, users, 'Scheme 06b: Reply Time Std-Dev σ (min)', '06b', output_dir, cmap='YlOrRd_r')
        self.create_heatmap(matrix_06_priority, users, 'Scheme 06c: Priority Score 1/(1+√(μ·σ))', '06c', output_dir, cmap='YlOrRd')
        
        # Scheme 07: Asymmetric reciprocity (signed difference, diverging)
        print("  Computing 07 (asymmetric reciprocity)...")
        matrix_07, _ = self.compute_07_asymmetry()
        self.create_heatmap(matrix_07, users, 'Scheme 07: Asymmetry A→B − B→A (+ = A chases B)', '07', output_dir, cmap='RdBu_r', diverging=True)
        
        # Save all matrices to npz
        np.savez(os.path.join(output_dir, 'scheme_matrices.npz'),
                 users=np.array(users),
                 m01=matrix_01,
                 m01b=self.normalize_receiver(matrix_01, users),
                 m01c=self.normalize_sender(matrix_01, users),
                 m01d=self.normalize_double(matrix_01, users),
                 m02=matrix_02, m02b=matrix_02_norm_r, m02c=matrix_02_norm_s, m02d=matrix_02_double,
                 m03=matrix_03, m03b=matrix_03_norm_r, m03c=matrix_03_norm_s, m03d=matrix_03_double,
                 m04=matrix_04, m04b=matrix_04_norm_r, m04c=matrix_04_norm_s, m04d=matrix_04_double,
                 m05=matrix_05, m05b=matrix_05_norm_r, m05c=matrix_05_norm_s, m05d=matrix_05_double,
                 m06_mean=matrix_06_mean, m06_std=matrix_06_std, m06_priority=matrix_06_priority,
                 m07=matrix_07)
        print(f"  💾 Saved matrices to {output_dir}/scheme_matrices.npz")
        
        print(f"\n✅ All matrices computed! Saved to: {output_dir}/")
    
    # ========== TIME-SERIES ANALYSIS ==========
    
    def analyze_time_series(self, output_dir='matrix'):
        """Generate time-series panoramas with 4-month segments (Jan-Apr, May-Aug, Sep-Dec)"""
        print("\n📅 Computing time-series analysis...\n")
        
        # Find data date range
        min_date = min(msg['timestamp'] for msg in self.messages)
        max_date = max(msg['timestamp'] for msg in self.messages)
        
        # Generate 4-month segments
        segment_defs = [(1, 4, 'Jan-Apr'), (5, 8, 'May-Aug'), (9, 12, 'Sep-Dec')]
        segments = []
        
        for year in range(min_date.year, max_date.year + 1):
            for start_month, end_month, label in segment_defs:
                seg_start = datetime(year, start_month, 1)
                if end_month == 12:
                    seg_end = datetime(year + 1, 1, 1)
                else:
                    seg_end = datetime(year, end_month + 1, 1)
                
                # Only include segments that overlap with data
                if seg_end <= min_date or seg_start > max_date:
                    continue
                
                segments.append((seg_start, seg_end, f"{year} {label}"))
        
        # Show segment info
        print(f"  Found {len(segments)} segments:")
        for seg_start, seg_end, label in segments:
            count = sum(1 for msg in self.messages if seg_start <= msg['timestamp'] < seg_end)
            print(f"    {label}: {count} messages")
        
        # Store originals
        original_messages = self.messages
        original_users = self.users
        all_users = sorted(original_users)  # Consistent user list across all segments
        
        matrices_02 = []  # raw
        matrices_03 = []  # raw
        matrices_04 = []  # raw
        matrices_05 = []  # raw
        labels = []
        
        for seg_start, seg_end, label in segments:
            seg_messages = [msg for msg in original_messages if seg_start <= msg['timestamp'] < seg_end]
            labels.append(label)
            
            n_u = len(all_users)
            if len(seg_messages) < 2:
                matrices_02.append(np.zeros((n_u, n_u)))
                matrices_03.append(np.zeros((n_u, n_u)))
                matrices_04.append(np.zeros((n_u, n_u)))
                matrices_05.append(np.zeros((n_u, n_u)))
                continue
            
            # Temporarily swap messages and users for this segment
            self.messages = seg_messages
            self.users = set(all_users)
            
            matrix_02, _ = self.compute_02_time_decay_immediate()
            matrices_02.append(matrix_02)
            
            matrix_03, _ = self.compute_03_time_decay_window()
            matrices_03.append(matrix_03)
            
            matrix_04, _ = self.compute_04_late_night()
            matrices_04.append(matrix_04)
            
            matrix_05, _ = self.compute_05_exclusivity()
            matrices_05.append(matrix_05)
        
        # Restore originals
        self.messages = original_messages
        self.users = original_users
        
        # Derive a/b/c/d variants for each scheme
        def derive_variants(raw_matrices):
            """Return (a, b, c, d) lists of matrices"""
            a, b, c, d = [], [], [], []
            for seg_idx, m in enumerate(raw_matrices):
                a.append(m)
                # Need segment messages for normalization
                seg_start, seg_end, _ = segments[seg_idx]
                seg_msgs = [msg for msg in original_messages if seg_start <= msg['timestamp'] < seg_end]
                self.messages = seg_msgs
                self.users = set(all_users)
                b.append(self.normalize_receiver(m, all_users))
                c.append(self.normalize_sender(m, all_users))
                d.append(self.normalize_double(m, all_users))
            self.messages = original_messages
            self.users = original_users
            return a, b, c, d
        
        m02a, m02b, m02c, m02d = derive_variants(matrices_02)
        m03a, m03b, m03c, m03d = derive_variants(matrices_03)
        m04a, m04b, m04c, m04d = derive_variants(matrices_04)
        m05a, m05b, m05c, m05d = derive_variants(matrices_05)
        
        # Save tensors (rank-3: segments x users x users)
        os.makedirs(output_dir, exist_ok=True)
        tensor_path = os.path.join(output_dir, 'time_series_tensors.npz')
        np.savez(tensor_path,
                 users=np.array(all_users),
                 labels=np.array(labels),
                 s02_a=np.stack(m02a), s02_b=np.stack(m02b), s02_c=np.stack(m02c), s02_d=np.stack(m02d),
                 s03_a=np.stack(m03a), s03_b=np.stack(m03b), s03_c=np.stack(m03c), s03_d=np.stack(m03d),
                 s04_a=np.stack(m04a), s04_b=np.stack(m04b), s04_c=np.stack(m04c), s04_d=np.stack(m04d),
                 s05_a=np.stack(m05a), s05_b=np.stack(m05b), s05_c=np.stack(m05c), s05_d=np.stack(m05d))
        print(f"\n  💾 Saved tensors to {tensor_path}")
        
        # Create 16 panoramas (4 schemes x 4 normalization variants)
        panoramas = [
            (m02a, 'Time Series 02-a — Scheme 02a: Time-Decay Immediate (Absolute)', 'time_series_02-a'),
            (m02b, 'Time Series 02-b — Scheme 02b: Time-Decay Immediate (Receiver-Norm)', 'time_series_02-b'),
            (m02c, 'Time Series 02-c — Scheme 02c: Time-Decay Immediate (Sender-Norm)', 'time_series_02-c'),
            (m02d, 'Time Series 02-d — Scheme 02d: Time-Decay Immediate (Double-Norm)', 'time_series_02-d'),
            (m03a, 'Time Series 03-a — Scheme 03a: Time-Decay 10-Window (Absolute)', 'time_series_03-a'),
            (m03b, 'Time Series 03-b — Scheme 03b: Time-Decay 10-Window (Receiver-Norm)', 'time_series_03-b'),
            (m03c, 'Time Series 03-c — Scheme 03c: Time-Decay 10-Window (Sender-Norm)', 'time_series_03-c'),
            (m03d, 'Time Series 03-d — Scheme 03d: Time-Decay 10-Window (Double-Norm)', 'time_series_03-d'),
            (m04a, 'Time Series 04-a — Scheme 04a: Late-Night + DM (Absolute)', 'time_series_04-a'),
            (m04b, 'Time Series 04-b — Scheme 04b: Late-Night + DM (Receiver-Norm)', 'time_series_04-b'),
            (m04c, 'Time Series 04-c — Scheme 04c: Late-Night + DM (Sender-Norm)', 'time_series_04-c'),
            (m04d, 'Time Series 04-d — Scheme 04d: Late-Night + DM (Double-Norm)', 'time_series_04-d'),
            (m05a, 'Time Series 05-a — Scheme 05a: Exclusivity Index (Absolute)', 'time_series_05-a'),
            (m05b, 'Time Series 05-b — Scheme 05b: Exclusivity Index (Receiver-Norm)', 'time_series_05-b'),
            (m05c, 'Time Series 05-c — Scheme 05c: Exclusivity Index (Sender-Norm)', 'time_series_05-c'),
            (m05d, 'Time Series 05-d — Scheme 05d: Exclusivity Index (Double-Norm)', 'time_series_05-d'),
        ]
        
        print()
        for mats, title, fname in panoramas:
            self._create_panorama(mats, all_users, labels, title, fname, output_dir)
    
    def _create_panorama(self, matrices, all_users, labels, title, filename, output_dir,
                         cmap='YlOrRd', use_log=True, diverging=False):
        """Create a panorama PNG with matrices in a 3-column grid layout"""
        n = len(matrices)
        n_users = len(all_users)
        ncols = 3
        nrows = -(-n // ncols)  # Ceiling division
        
        # Size each subplot appropriately
        cell_w = max(4.0, n_users * 0.32)
        cell_h = max(4.5, n_users * 0.34)
        
        # Use gridspec to reserve a thin column for the colorbar on the right
        fig = plt.figure(figsize=(cell_w * ncols + 3, cell_h * nrows + 1.5))
        gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[1] * ncols + [0.05], wspace=0.35, hspace=0.4)
        
        axes_flat = []
        for r in range(nrows):
            for c in range(ncols):
                axes_flat.append(fig.add_subplot(gs[r, c]))
        cbar_ax = fig.add_subplot(gs[:, ncols])
        
        # Compute display values and color range
        if use_log:
            display_matrices = [np.log1p(m) for m in matrices]
        else:
            display_matrices = [m.copy() for m in matrices]
        
        all_vals = np.concatenate([dm.flatten() for dm in display_matrices])
        if diverging:
            abs_max = max(abs(all_vals.min()), abs(all_vals.max()))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = all_vals.min(), all_vals.max()
        
        im = None
        for idx in range(len(axes_flat)):
            ax = axes_flat[idx]
            
            if idx >= n:
                ax.set_visible(False)
                continue
            
            matrix_display = display_matrices[idx]
            
            im = ax.imshow(matrix_display, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            
            ax.set_xticks(range(n_users))
            ax.set_yticks(range(n_users))
            ax.set_xticklabels(all_users, rotation=90, ha='center', fontsize=6)
            ax.set_yticklabels(all_users, fontsize=6)
            
            ax.set_title(labels[idx], fontsize=9, fontweight='bold')
        
        cbar_label = 'Difference' if diverging else ('Score (log scale)' if use_log else 'Score')
        fig.colorbar(im, cax=cbar_ax, label=cbar_label)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        filepath = os.path.join(output_dir, f'{filename}.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved {filepath}")
    
    # ========== DERIVATIVE MODE ==========
    
    def analyze_derivative(self, output_dir='matrix'):
        """Compute differences between adjacent time segments from saved tensors"""
        tensor_path = os.path.join(output_dir, 'time_series_tensors.npz')
        
        if not os.path.exists(tensor_path):
            print(f"❌ Error: {tensor_path} not found. Run --time-series first.")
            sys.exit(1)
        
        print("\n📈 Computing derivative analysis...\n")
        
        data = np.load(tensor_path, allow_pickle=True)
        all_users = list(data['users'])
        seg_labels = list(data['labels'])
        
        # Compute derivative labels (e.g. "2023 May-Aug → Sep-Dec")
        deriv_labels = []
        for i in range(len(seg_labels) - 1):
            # Shorten: take year from first, period from both
            parts_a = seg_labels[i].split(' ', 1)
            parts_b = seg_labels[i + 1].split(' ', 1)
            deriv_labels.append(f"{parts_a[0]} {parts_a[1]}\n→ {parts_b[0]} {parts_b[1]}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        variants = [
            ('s02_a', 'Derivative 02-a — Δ02a: Time-Decay Immediate (Absolute)', 'derivative_02-a'),
            ('s02_b', 'Derivative 02-b — Δ02b: Time-Decay Immediate (Receiver-Norm)', 'derivative_02-b'),
            ('s02_c', 'Derivative 02-c — Δ02c: Time-Decay Immediate (Sender-Norm)', 'derivative_02-c'),
            ('s02_d', 'Derivative 02-d — Δ02d: Time-Decay Immediate (Double-Norm)', 'derivative_02-d'),
            ('s03_a', 'Derivative 03-a — Δ03a: Time-Decay 10-Window (Absolute)', 'derivative_03-a'),
            ('s03_b', 'Derivative 03-b — Δ03b: Time-Decay 10-Window (Receiver-Norm)', 'derivative_03-b'),
            ('s03_c', 'Derivative 03-c — Δ03c: Time-Decay 10-Window (Sender-Norm)', 'derivative_03-c'),
            ('s03_d', 'Derivative 03-d — Δ03d: Time-Decay 10-Window (Double-Norm)', 'derivative_03-d'),
            ('s04_a', 'Derivative 04-a — Δ04a: Late-Night + DM (Absolute)', 'derivative_04-a'),
            ('s04_b', 'Derivative 04-b — Δ04b: Late-Night + DM (Receiver-Norm)', 'derivative_04-b'),
            ('s04_c', 'Derivative 04-c — Δ04c: Late-Night + DM (Sender-Norm)', 'derivative_04-c'),
            ('s04_d', 'Derivative 04-d — Δ04d: Late-Night + DM (Double-Norm)', 'derivative_04-d'),
            ('s05_a', 'Derivative 05-a — Δ05a: Exclusivity Index (Absolute)', 'derivative_05-a'),
            ('s05_b', 'Derivative 05-b — Δ05b: Exclusivity Index (Receiver-Norm)', 'derivative_05-b'),
            ('s05_c', 'Derivative 05-c — Δ05c: Exclusivity Index (Sender-Norm)', 'derivative_05-c'),
            ('s05_d', 'Derivative 05-d — Δ05d: Exclusivity Index (Double-Norm)', 'derivative_05-d'),
        ]
        
        for key, title, fname in variants:
            tensor = data[key]  # shape: (n_segments, n_users, n_users)
            # Compute differences between adjacent segments
            diffs = [tensor[i + 1] - tensor[i] for i in range(tensor.shape[0] - 1)]
            self._create_panorama(diffs, all_users, deriv_labels, title, fname, output_dir,
                                  cmap='RdBu_r', use_log=False, diverging=True)
        
        print(f"\n✅ All derivative matrices computed! Saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze adjacency matrices with multiple schemes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python adjacency_matrix_analysis.py alfafa/alfafa_20260219.txt
  python adjacency_matrix_analysis.py alfafa/alfafa_20260219.txt --exclude-bottom 5
        """
    )
    
    parser.add_argument('input_file', help='Path to WhatsApp chat export (.txt file)')
    parser.add_argument('--exclude-bottom', type=int, default=3, metavar='N',
                       help='Exclude N users with lowest message counts (default: 3)')
    parser.add_argument('--time-series', action='store_true',
                       help='Generate time-series panoramas (4-month segments)')
    parser.add_argument('--derivative', action='store_true',
                       help='Generate derivative panoramas (requires --time-series data)')
    parser.add_argument('--window', type=int, default=10, metavar='N',
                       help='Window size for schemes 03-06 (default: 10)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
    
    input_dir = os.path.dirname(args.input_file)
    if not input_dir:
        input_dir = "."
    
    if args.window != 10:
        output_dir = os.path.join(input_dir, f"matrix_{args.window}")
    else:
        output_dir = os.path.join(input_dir, "matrix")
    
    try:
        analyzer = AdjacencyMatrixAnalyzer(args.input_file, exclude_bottom_n=args.exclude_bottom, window_size=args.window)
        
        print(f"📊 Analyzing: {args.input_file}")
        print(f"📁 Matrices will be saved to: {output_dir}/")
        print()
        
        if args.derivative:
            analyzer.analyze_derivative(output_dir=output_dir)
        elif args.time_series:
            analyzer.analyze_time_series(output_dir=output_dir)
        else:
            analyzer.analyze_all_schemes(output_dir=output_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
