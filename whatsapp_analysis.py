import re
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from scipy import stats
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

class ChatAnalyzer:
    def __init__(self, input_file):
        self.input_file = input_file
        self.messages = []
        self.users = set()
        self.parse_messages()
    
    def parse_messages(self):
        """Parse messages from WhatsApp export file"""
        # Support both formats: [DD/M/YYYY, HH:MM:SS] and [DD/MM/YYYY, HH:MM:SS am/pm]
        pattern = r'\[(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}:\d{2}(?:\s+(?:am|pm|AM|PM))?)\] ([^:]+): (.*)'
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by timestamp pattern to handle multi-line messages
        lines = content.split('\n')
        current_message = None
        
        for line in lines:
            match = re.match(pattern, line)
            if match:
                # Save previous message if exists
                if current_message:
                    self.messages.append(current_message)
                
                date_str, time_str, sender, text = match.groups()
                
                # Parse time - handle both 24-hour and 12-hour formats
                time_str = time_str.strip()
                try:
                    if any(x in time_str.lower() for x in ['am', 'pm']):
                        # 12-hour format with am/pm
                        timestamp = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %I:%M:%S %p")
                    else:
                        # 24-hour format
                        timestamp = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M:%S")
                except ValueError:
                    # Fallback: try alternative format
                    try:
                        timestamp = datetime.strptime(f"{date_str} {time_str.lower().replace(' am', ' AM').replace(' pm', ' PM')}", "%d/%m/%Y %I:%M:%S %p")
                    except:
                        continue
                
                current_message = {
                    'timestamp': timestamp,
                    'sender': sender,
                    'text': text,
                    'date': timestamp.date(),
                    'hour': timestamp.hour,
                    'weekday': timestamp.weekday(),
                    'date_str': date_str
                }
                self.users.add(sender)
            else:
                # Multi-line message continuation
                if current_message and line.strip():
                    current_message['text'] += '\n' + line
        
        # Don't forget the last message
        if current_message:
            self.messages.append(current_message)
    
    def analyze_user_stats(self):
        """Analyze per-user statistics"""
        stats = {}
        for user in self.users:
            user_messages = [m for m in self.messages if m['sender'] == user]
            total_chars = sum(len(m['text']) for m in user_messages)
            total_words = sum(len(m['text'].split()) for m in user_messages)
            
            stats[user] = {
                'message_count': len(user_messages),
                'total_chars': total_chars,
                'total_words': total_words,
                'avg_msg_length': total_chars / len(user_messages) if user_messages else 0,
                'avg_word_per_msg': total_words / len(user_messages) if user_messages else 0,
                'percentage': (len(user_messages) / len(self.messages) * 100) if self.messages else 0
            }
        
        return sorted(stats.items(), key=lambda x: x[1]['message_count'], reverse=True)
    
    def analyze_temporal(self):
        """Analyze temporal patterns"""
        hourly = Counter(m['hour'] for m in self.messages)
        daily = Counter(m['weekday'] for m in self.messages)
        date_wise = Counter(m['date'] for m in self.messages)
        
        return {
            'hourly': dict(sorted(hourly.items())),
            'weekday': dict(sorted(daily.items())),
            'date_wise': dict(sorted(date_wise.items()))
        }
    
    def analyze_messages(self):
        """Analyze message characteristics"""
        stats = {
            'text_messages': 0,
            'image_messages': 0,
            'sticker_messages': 0,
            'link_messages': 0,
            'total_messages': len(self.messages)
        }
        
        for msg in self.messages:
            text = msg['text'].lower()
            if 'image omitted' in text:
                stats['image_messages'] += 1
            elif 'sticker omitted' in text:
                stats['sticker_messages'] += 1
            elif 'http' in text or 'www.' in text:
                stats['link_messages'] += 1
            else:
                stats['text_messages'] += 1
        
        return stats
    
    def analyze_mentions(self):
        """Analyze @mentions with bidirectional relationships"""
        # More flexible mention pattern to capture mentions
        mention_pattern = r'@[⁨]?([^⁩\n]+?)[⁩]?(?:\s|$)'
        
        mention_matrix = defaultdict(lambda: defaultdict(int))  # who @s whom
        total_mentions = Counter()
        mention_details = []
        
        for msg in self.messages:
            text = msg['text']
            found_mentions = re.findall(mention_pattern, text)
            
            for mention in found_mentions:
                mention_clean = mention.strip()
                if mention_clean:
                    mention_matrix[msg['sender']][mention_clean] += 1
                    total_mentions[mention_clean] += 1
                    mention_details.append({
                        'from': msg['sender'],
                        'to': mention_clean,
                        'timestamp': msg['timestamp']
                    })
        
        return {
            'total_mentions': dict(total_mentions),
            'mention_matrix': dict(mention_matrix),
            'mention_details': mention_details,
            'total_mention_instances': len(mention_details)
        }
    
    def analyze_time_gaps(self):
        """Analyze time gaps between consecutive messages to identify conversation boundaries"""
        gaps = []
        gap_details = []
        
        for i in range(len(self.messages) - 1):
            current = self.messages[i]
            next_msg = self.messages[i + 1]
            gap_minutes = (next_msg['timestamp'] - current['timestamp']).total_seconds() / 60
            gaps.append(gap_minutes)
            gap_details.append({
                'gap_minutes': gap_minutes,
                'from_user': current['sender'],
                'to_user': next_msg['sender'],
                'timestamp': current['timestamp'],
                'is_same_user': current['sender'] == next_msg['sender']
            })
        
        if not gaps:
            return {'gaps': [], 'gap_details': []}
        
        # Statistical analysis
        mean_gap = sum(gaps) / len(gaps)
        variance = sum((x - mean_gap) ** 2 for x in gaps) / len(gaps)
        std_dev = variance ** 0.5
        
        # Use more intelligent threshold: 95th percentile (gaps larger than 95% are rare)
        # This naturally identifies when conversations likely end
        sorted_gaps = sorted(gaps)
        percentile_90 = sorted_gaps[int(0.90 * len(gaps))]
        percentile_95 = sorted_gaps[int(0.95 * len(gaps))]
        threshold = percentile_95  # Gaps > 95th percentile mark conversation boundaries
        
        return {
            'gaps': gaps,
            'gap_details': gap_details,
            'mean_gap': mean_gap,
            'std_dev': std_dev,
            'threshold': threshold,
            'percentile_25': sorted(gaps)[len(gaps) // 4],
            'percentile_50': sorted(gaps)[len(gaps) // 2],
            'percentile_75': sorted(gaps)[3 * len(gaps) // 4],
            'percentile_90': sorted(gaps)[int(0.90 * len(gaps))],
            'percentile_95': sorted(gaps)[int(0.95 * len(gaps))]
        }
    
    def analyze_conversations(self):
        """Identify and analyze conversation threads based on time gaps"""
        gap_data = self.analyze_time_gaps()
        if not gap_data['gap_details']:
            return {}
        
        threshold = gap_data['threshold']
        conversations = []
        current_convo = {'start_idx': 0, 'messages': 1, 'participants': {self.messages[0]['sender']}}
        
        for i, gap_detail in enumerate(gap_data['gap_details']):
            if gap_detail['gap_minutes'] > threshold:
                # End current conversation
                current_convo['end_idx'] = i
                current_convo['duration_minutes'] = gap_detail['gap_minutes']
                conversations.append(current_convo)
                # Start new conversation
                current_convo = {'start_idx': i + 1, 'messages': 1, 'participants': {self.messages[i + 1]['sender']}}
            else:
                current_convo['messages'] += 1
                current_convo['participants'].add(self.messages[i + 1]['sender'])
        
        # Add last conversation
        current_convo['end_idx'] = len(self.messages) - 1
        conversations.append(current_convo)
        
        # Calculate stats
        convo_stats = {
            'total_conversations': len(conversations),
            'avg_messages_per_convo': sum(c['messages'] for c in conversations) / len(conversations),
            'max_messages_in_convo': max(c['messages'] for c in conversations),
            'avg_participants_per_convo': sum(len(c['participants']) for c in conversations) / len(conversations),
            'threshold_used': threshold
        }
        
        return {'stats': convo_stats, 'conversations': conversations}
    
    def analyze_response_times_in_conversation(self):
        """Analyze response times within conversations (excluding large gaps)"""
        gap_data = self.analyze_time_gaps()
        if not gap_data['gap_details']:
            return {}
        
        # Use a more conservative threshold to capture real responses better
        threshold = 65  # Use 65 minutes as per investigation
        response_times = []
        response_by_user = defaultdict(list)
        
        for gap_detail in gap_data['gap_details']:
            # Only consider positive gaps within same conversation
            if gap_detail['gap_minutes'] > 0 and gap_detail['gap_minutes'] < threshold:
                response_times.append(gap_detail['gap_minutes'])
                response_by_user[gap_detail['to_user']].append(gap_detail['gap_minutes'])
        
        if not response_times:
            return {}
        
        # Overall stats
        overall = {
            'avg_response_time': sum(response_times) / len(response_times),
            'median_response_time': sorted(response_times)[len(response_times) // 2],
            'max_response_time': max(response_times),
            'min_response_time': min(response_times),
            'total_responses': len(response_times)
        }
        
        # Per-user stats
        user_response_stats = {}
        for user in response_by_user:
            times = response_by_user[user]
            if times:  # Only add if user has responses
                user_response_stats[user] = {
                    'avg_response_time': sum(times) / len(times),
                    'responses': len(times),
                    'median_response_time': sorted(times)[len(times) // 2]
                }
        
        return {'overall': overall, 'by_user': user_response_stats}
    
    def analyze_language(self):
        """Analyze language usage patterns"""
        stats = {
            'english_messages': 0,
            'chinese_messages': 0,
            'emoji_messages': 0,
            'mixed_messages': 0
        }
        
        # Simple heuristic: check for Chinese characters
        for msg in self.messages:
            text = msg['text']
            has_english = bool(re.search(r'[a-zA-Z]', text))
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
            has_emoji = bool(re.search(r'[\U0001F300-\U0001F9FF]', text))
            
            if has_emoji:
                stats['emoji_messages'] += 1
            
            if has_english and has_chinese:
                stats['mixed_messages'] += 1
            elif has_english:
                stats['english_messages'] += 1
            elif has_chinese:
                stats['chinese_messages'] += 1
        
        return stats
    
    def analyze_engagement(self):
        """Analyze overall engagement metrics"""
        if not self.messages:
            return {}
        
        first_msg = self.messages[0]['timestamp']
        last_msg = self.messages[-1]['timestamp']
        time_span = (last_msg - first_msg).days
        
        return {
            'first_message': first_msg,
            'last_message': last_msg,
            'days_active': time_span if time_span > 0 else 1,
            'messages_per_day': len(self.messages) / (time_span if time_span > 0 else 1),
            'unique_users': len(self.users),
            'total_messages': len(self.messages)
        }
    
    def analyze_user_burstiness(self):
        """Analyze how often users send multiple consecutive messages (burstiness)"""
        user_burstiness = defaultdict(lambda: {'bursts': 0, 'burst_sizes': [], 'total_messages': 0})
        
        for i, msg in enumerate(self.messages):
            user = msg['sender']
            user_burstiness[user]['total_messages'] += 1
            
            # Check if next message is also from same user
            if i < len(self.messages) - 1 and self.messages[i + 1]['sender'] == user:
                burst_size = 1
                j = i + 1
                while j < len(self.messages) and self.messages[j]['sender'] == user:
                    burst_size += 1
                    j += 1
                
                if burst_size > 1:
                    user_burstiness[user]['bursts'] += 1
                    user_burstiness[user]['burst_sizes'].append(burst_size)
        
        # Calculate metrics
        burst_stats = {}
        for user, data in user_burstiness.items():
            total = data['total_messages']
            burst_count = data['bursts']
            burst_stats[user] = {
                'burst_frequency': (burst_count / total * 100) if total > 0 else 0,
                'avg_burst_size': sum(data['burst_sizes']) / len(data['burst_sizes']) if data['burst_sizes'] else 0,
                'burst_count': burst_count,
                'total_messages': total
            }
        
        return burst_stats
    
    def analyze_user_networks(self):
        """Analyze conversation networks and user interactions"""
        G = nx.DiGraph()
        
        for msg in self.messages:
            G.add_node(msg['sender'])
        
        # Add edges based on replies (message followed by different user)
        interaction_pairs = defaultdict(int)
        for i, msg in enumerate(self.messages[:-1]):
            next_msg = self.messages[i + 1]
            if msg['sender'] != next_msg['sender']:
                interaction_pairs[(msg['sender'], next_msg['sender'])] += 1
                G.add_edge(msg['sender'], next_msg['sender'], weight=interaction_pairs[(msg['sender'], next_msg['sender'])])
        
        return {
            'graph': G,
            'interaction_pairs': dict(interaction_pairs),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G.to_undirected()) if len(G) > 0 else 0
        }
    
    def analyze_engagement_trends(self):
        """Analyze how engagement changes over time"""
        daily_stats = defaultdict(lambda: {
            'messages': 0,
            'users': set(),
            'chars': 0
        })
        
        for msg in self.messages:
            date = msg['date']
            daily_stats[date]['messages'] += 1
            daily_stats[date]['users'].add(msg['sender'])
            daily_stats[date]['chars'] += len(msg['text'])
        
        # Convert to time series
        dates = sorted(daily_stats.keys())
        messages_per_day = [daily_stats[d]['messages'] for d in dates]
        users_per_day = [len(daily_stats[d]['users']) for d in dates]
        chars_per_day = [daily_stats[d]['chars'] for d in dates]
        
        return {
            'dates': dates,
            'messages_per_day': messages_per_day,
            'users_per_day': users_per_day,
            'chars_per_day': chars_per_day,
            'trend_messages': np.polyfit(range(len(messages_per_day)), messages_per_day, 2) if len(messages_per_day) > 2 else None
        }
    
    def analyze_user_roles(self):
        """Classify users by their role in conversations"""
        roles = {}
        
        for user in self.users:
            user_msgs = [m for m in self.messages if m['sender'] == user]
            
            # Calculate metrics
            total_msgs = len(user_msgs)
            avg_length = sum(len(m['text']) for m in user_msgs) / total_msgs if user_msgs else 0
            
            # Find who this user responds to most
            responses_to = defaultdict(int)
            for i, msg in enumerate(self.messages):
                if msg['sender'] == user and i > 0:
                    prev_sender = self.messages[i-1]['sender']
                    if prev_sender != user:
                        responses_to[prev_sender] += 1
            
            # Who responds to this user most
            responses_from = defaultdict(int)
            for i, msg in enumerate(self.messages):
                if self.messages[i-1]['sender'] == user and msg['sender'] != user and i > 0:
                    responses_from[msg['sender']] += 1
            
            # Classify role
            if total_msgs > 0:
                if avg_length > np.percentile([len(m['text']) for m in self.messages], 75):
                    role_type = "Thoughtful"  # Long messages
                elif total_msgs > np.percentile([len([m for m in self.messages if m['sender'] == u]) for u in self.users], 75):
                    role_type = "Active"  # Many messages
                elif len(responses_from) > len(responses_to):
                    role_type = "Initiator"  # Others respond more to them
                else:
                    role_type = "Responder"  # They respond more
            else:
                role_type = "Inactive"
            
            roles[user] = {
                'role': role_type,
                'total_messages': total_msgs,
                'avg_length': avg_length,
                'main_respondents': dict(sorted(responses_from.items(), key=lambda x: x[1], reverse=True)[:3]),
                'main_responds_to': dict(sorted(responses_to.items(), key=lambda x: x[1], reverse=True)[:3])
            }
        
        return roles
    
    def analyze_message_entropy(self):
        """Analyze message variability and patterns"""
        entropy_by_user = {}
        
        for user in self.users:
            user_msgs = [m for m in self.messages if m['sender'] == user]
            if not user_msgs:
                continue
            
            # Message length distribution entropy
            lengths = [len(m['text']) for m in user_msgs]
            if lengths:
                # Normalize and calculate entropy
                hist, _ = np.histogram(lengths, bins=min(10, len(set(lengths))))
                hist = hist / hist.sum()
                entropy = -np.sum([p * np.log2(p + 1e-10) for p in hist])
                
                entropy_by_user[user] = {
                    'message_entropy': entropy,  # Higher = more variable message lengths
                    'consistency': 1 - (entropy / np.log2(len(hist))),  # 0-1 scale
                    'avg_length': np.mean(lengths),
                    'std_length': np.std(lengths)
                }
        
        return entropy_by_user
    
    def analyze_emoji_sentiment(self):
        """Analyze sentiment through emoji usage patterns"""
        emoji_positivity = {
            '😊': 2, '😄': 2, '😍': 3, '🥰': 3, '😘': 2, '💕': 2, '❤️': 3,
            '😢': -2, '😭': -3, '😡': -3, '😱': -1, '😠': -2,
            '😂': 2, '🤣': 2, '😆': 2,  # Laughing
            '🎉': 2, '🙌': 2, '👏': 2,  # Celebration
            '💪': 1, '✨': 1, '🌟': 1  # Positive
        }
        
        sentiment_by_user = {}
        
        for user in self.users:
            user_msgs = [m for m in self.messages if m['sender'] == user]
            scores = []
            emoji_counts = Counter()
            
            for msg in user_msgs:
                text = msg['text']
                # Extract emojis (simple approach)
                for emoji, score in emoji_positivity.items():
                    if emoji in text:
                        emoji_counts[emoji] += text.count(emoji)
                        scores.extend([score] * text.count(emoji))
            
            sentiment_score = np.mean(scores) if scores else 0
            emoji_total = sum(emoji_counts.values())
            
            sentiment_by_user[user] = {
                'sentiment_score': sentiment_score,  # -3 to 3 scale
                'emoji_count': emoji_total,
                'emoji_per_message': emoji_total / len(user_msgs) if user_msgs else 0,
                'top_emojis': dict(emoji_counts.most_common(5))
            }
        
        return sentiment_by_user
    
    def analyze_conversation_starters(self):
        """Identify who typically starts conversations"""
        starters = defaultdict(int)
        current_convo_start = True
        threshold = self.analyze_time_gaps()['threshold']
        
        for i, msg in enumerate(self.messages):
            if i == 0:
                starters[msg['sender']] += 1
            elif current_convo_start:
                starters[msg['sender']] += 1
                current_convo_start = False
            
            # Check if next gap is large (new conversation)
            if i < len(self.messages) - 1:
                gap = (self.messages[i + 1]['timestamp'] - msg['timestamp']).total_seconds() / 60
                if gap > threshold:
                    current_convo_start = True
        
        return dict(starters)
    
    def generate_report(self, output_file):
        """Generate markdown report"""
        report = []
        
        # Title and metadata
        report.append("# Chat Analysis Report: Alfafa Group\n")
        report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # Overall engagement
        engagement = self.analyze_engagement()
        report.append("## Overview\n")
        report.append(f"- **Total Messages**: {engagement['total_messages']}\n")
        report.append(f"- **Unique Participants**: {engagement['unique_users']}\n")
        report.append(f"- **Active Days**: {engagement['days_active']}\n")
        report.append(f"- **Messages per Day**: {engagement['messages_per_day']:.2f}\n")
        report.append(f"- **Date Range**: {engagement['first_message'].date()} to {engagement['last_message'].date()}\n")
        
        # User statistics
        report.append("\n## User Statistics\n")
        user_stats = self.analyze_user_stats()
        report.append("| User | Messages | % of Activity | Avg Length | Total Chars |\n")
        report.append("|------|----------|---------------|-----------|----|\n")
        for user, stats in user_stats:
            report.append(f"| {user} | {stats['message_count']} | {stats['percentage']:.1f}% | {stats['avg_msg_length']:.1f} | {stats['total_chars']} |\n")
        
        # Message types
        report.append("\n## Message Composition\n")
        msg_stats = self.analyze_messages()
        total = msg_stats['total_messages']
        report.append(f"- **Text Messages**: {msg_stats['text_messages']} ({msg_stats['text_messages']/total*100:.1f}%)\n")
        report.append(f"- **Images**: {msg_stats['image_messages']} ({msg_stats['image_messages']/total*100:.1f}%)\n")
        report.append(f"- **Stickers**: {msg_stats['sticker_messages']} ({msg_stats['sticker_messages']/total*100:.1f}%)\n")
        report.append(f"- **Links**: {msg_stats['link_messages']} ({msg_stats['link_messages']/total*100:.1f}%)\n")
        
        # Language analysis
        report.append("\n## Language Patterns\n")
        lang_stats = self.analyze_language()
        report.append(f"- **Emoji Usage**: {lang_stats['emoji_messages']} messages ({lang_stats['emoji_messages']/len(self.messages)*100:.1f}%)\n")
        report.append(f"- **English Only**: {lang_stats['english_messages']} messages ({lang_stats['english_messages']/len(self.messages)*100:.1f}%)\n")
        report.append(f"- **Chinese Only**: {lang_stats['chinese_messages']} messages ({lang_stats['chinese_messages']/len(self.messages)*100:.1f}%)\n")
        report.append(f"- **Mixed Language**: {lang_stats['mixed_messages']} messages ({lang_stats['mixed_messages']/len(self.messages)*100:.1f}%)\n")
        
        # Temporal analysis
        report.append("\n## Temporal Patterns\n")
        temporal = self.analyze_temporal()
        
        report.append("### Messages by Hour of Day\n")
        report.append("| Hour | Count |\n")
        report.append("|------|-------|\n")
        for hour in sorted(temporal['hourly'].keys()):
            report.append(f"| {hour:02d}:00 | {temporal['hourly'][hour]} |\n")
        
        report.append("\n### Messages by Weekday\n")
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        report.append("| Day | Count |\n")
        report.append("|-----|-------|\n")
        for day in range(7):
            count = temporal['weekday'].get(day, 0)
            report.append(f"| {weekdays[day]} | {count} |\n")
        
        # Time gap analysis
        report.append("\n## Conversation Flow Analysis\n")
        gap_stats = self.analyze_time_gaps()
        report.append(f"### Time Gap Statistics (between consecutive messages)\n")
        report.append(f"- **Mean Gap**: {gap_stats['mean_gap']:.2f} minutes\n")
        report.append(f"- **Std Dev**: {gap_stats['std_dev']:.2f} minutes\n")
        report.append(f"- **25th Percentile**: {gap_stats['percentile_25']:.2f} minutes\n")
        report.append(f"- **Median (50th)**: {gap_stats['percentile_50']:.2f} minutes\n")
        report.append(f"- **75th Percentile**: {gap_stats['percentile_75']:.2f} minutes\n")
        report.append(f"- **90th Percentile**: {gap_stats['percentile_90']:.2f} minutes\n")
        report.append(f"- **95th Percentile**: {gap_stats['percentile_95']:.2f} minutes\n")
        report.append(f"- **Conversation Boundary Threshold**: {gap_stats['threshold']:.2f} minutes (~{gap_stats['threshold']/60:.1f} hours)\n")
        
        # Conversation identification
        report.append("\n### Conversation Threads\n")
        convo_data = self.analyze_conversations()
        convo_stats = convo_data['stats']
        report.append(f"- **Total Conversations Identified**: {convo_stats['total_conversations']}\n")
        report.append(f"- **Avg Messages per Conversation**: {convo_stats['avg_messages_per_convo']:.1f}\n")
        report.append(f"- **Max Messages in Single Conversation**: {convo_stats['max_messages_in_convo']}\n")
        report.append(f"- **Avg Participants per Conversation**: {convo_stats['avg_participants_per_convo']:.1f}\n")
        
        # Response time analysis
        report.append("\n### Response Times Within Conversations\n")
        response_stats = self.analyze_response_times_in_conversation()
        if response_stats:
            overall = response_stats['overall']
            report.append(f"#### Overall\n")
            report.append(f"- **Average Response Time**: {overall['avg_response_time']:.1f} minutes\n")
            report.append(f"- **Median Response Time**: {overall['median_response_time']:.1f} minutes\n")
            report.append(f"- **Min Response Time**: {overall['min_response_time']:.2f} minutes\n")
            report.append(f"- **Max Response Time**: {overall['max_response_time']:.1f} minutes\n")
            report.append(f"- **Total Response Instances**: {overall['total_responses']}\n")
            
            report.append(f"\n#### Top 15 Fastest Responders (avg response time)\n")
            report.append("| User | Avg Response (min) | Responses |\n")
            report.append("|------|-------------------|----------|\n")
            user_response = response_stats['by_user']
            for user in sorted(user_response.keys(), key=lambda x: user_response[x]['avg_response_time'])[:15]:
                stats = user_response[user]
                report.append(f"| {user} | {stats['avg_response_time']:.1f} | {stats['responses']} |\n")
        
        # @mention analysis - bidirectional
        report.append("\n## @Mention Relationships\n")
        mentions = self.analyze_mentions()
        if mentions['total_mention_instances'] > 0:
            report.append(f"- **Total @mentions**: {mentions['total_mention_instances']}\n")
            report.append(f"- **Unique users mentioned**: {len(mentions['total_mentions'])}\n\n")
            
            report.append(f"### Most Mentioned Users\n")
            report.append("| User | Times Mentioned |\n")
            report.append("|------|----------------|\n")
            for user, count in sorted(mentions['total_mentions'].items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"| {user} | {count} |\n")
            
            report.append(f"\n### @mention Matrix (Top Interactions)\n")
            report.append("| From User | To User | Count |\n")
            report.append("|-----------|---------|-------|\n")
            
            # Flatten and sort the matrix
            all_pairs = []
            for from_user, targets in mentions['mention_matrix'].items():
                for to_user, count in targets.items():
                    all_pairs.append((from_user, to_user, count))
            
            for from_user, to_user, count in sorted(all_pairs, key=lambda x: x[2], reverse=True)[:15]:
                report.append(f"| {from_user} | {to_user} | {count} |\n")
        else:
            report.append("No @mentions found in messages.\n")
        
        # Footer
        report.append("\n---\n")
        report.append("*This report analyzes message metadata and patterns. Personal reactions and media content details are not included as per analysis scope.*\n")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(''.join(report))
        
        print(f"✓ Report generated: {output_file}")
    
    def analyze_word_frequency(self):
        """Analyze top words used in the group"""
        from collections import Counter
        import re
        
        # Common words to exclude
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                     'that', 'this', 'it', 'its', 'he', 'she', 'they', 'we', 'you', 'i',
                     'image', 'omitted', 'sticker', 'http', 'www', 'url', 'link'}
        
        word_counts = Counter()
        
        for msg in self.messages:
            # Extract only text messages
            if 'image omitted' not in msg['text'].lower() and 'sticker omitted' not in msg['text'].lower():
                # Simple tokenization
                words = re.findall(r'\b[a-z\u4e00-\u9fff]+\b', msg['text'].lower())
                for word in words:
                    if word not in stopwords and len(word) > 2:
                        word_counts[word] += 1
        
        return dict(word_counts.most_common(50))
    
    def analyze_message_length_distribution(self):
        """Analyze message length patterns by user"""
        length_stats = {}
        
        for user in self.users:
            user_msgs = [m for m in self.messages if m['sender'] == user]
            lengths = [len(m['text']) for m in user_msgs]
            
            if lengths:
                length_stats[user] = {
                    'avg_length': np.mean(lengths),
                    'median_length': np.median(lengths),
                    'std_length': np.std(lengths),
                    'min_length': min(lengths),
                    'max_length': max(lengths),
                    'total_chars': sum(lengths),
                    'message_count': len(user_msgs)
                }
        
        return length_stats
    
    def analyze_peak_hours(self):
        """Analyze peak activity hours and times"""
        hourly_stats = defaultdict(lambda: {'messages': 0, 'users': set(), 'avg_length': []})
        
        for msg in self.messages:
            hour = msg['hour']
            hourly_stats[hour]['messages'] += 1
            hourly_stats[hour]['users'].add(msg['sender'])
            hourly_stats[hour]['avg_length'].append(len(msg['text']))
        
        # Calculate metrics
        peak_hours = []
        for hour in sorted(hourly_stats.keys()):
            data = hourly_stats[hour]
            peak_hours.append({
                'hour': hour,
                'messages': data['messages'],
                'unique_users': len(data['users']),
                'avg_msg_length': np.mean(data['avg_length']) if data['avg_length'] else 0,
                'total_chars': sum(data['avg_length'])
            })
        
        return peak_hours
    
    def analyze_user_dominance(self):
        """Calculate user dominance metrics"""
        dominance = {}
        total_chars = sum(len(m['text']) for m in self.messages)
        total_msgs = len(self.messages)
        
        for user in self.users:
            user_msgs = [m for m in self.messages if m['sender'] == user]
            user_chars = sum(len(m['text']) for m in user_msgs)
            
            dominance[user] = {
                'message_share': len(user_msgs) / total_msgs * 100,
                'character_share': user_chars / total_chars * 100,
                'influence_score': (len(user_msgs) / total_msgs * 100 + user_chars / total_chars * 100) / 2
            }
        
        return dominance
    
    def analyze_conversation_participation(self):
        """Calculate participation rates in conversations"""
        gap_data = self.analyze_time_gaps()
        threshold = gap_data['threshold']
        
        user_conversations = defaultdict(int)
        current_participants = set()
        
        for i, msg in enumerate(self.messages):
            current_participants.add(msg['sender'])
            
            # Check if next message starts new conversation
            if i < len(self.messages) - 1:
                next_msg = self.messages[i + 1]
                gap = (next_msg['timestamp'] - msg['timestamp']).total_seconds() / 60
                
                if gap > threshold:
                    # End of conversation
                    for user in current_participants:
                        user_conversations[user] += 1
                    current_participants = set()
        
        # Don't forget last conversation
        for user in current_participants:
            user_conversations[user] += 1
        
        return dict(user_conversations)
    
    def analyze_intertemporal_patterns(self):
        """Analyze how user behavior changes over time"""
        # Divide into quarters
        first_msg = self.messages[0]['timestamp']
        last_msg = self.messages[-1]['timestamp']
        date_range = (last_msg - first_msg).days
        quarter_length = date_range / 4
        
        quarters = []
        for q in range(4):
            start_date = first_msg + timedelta(days=q * quarter_length)
            end_date = first_msg + timedelta(days=(q + 1) * quarter_length)
            quarters.append({
                'quarter': q + 1,
                'start': start_date,
                'end': end_date,
                'users': defaultdict(lambda: {'messages': 0, 'chars': 0})
            })
        
        # Assign messages to quarters
        for msg in self.messages:
            msg_time = msg['timestamp']
            q_idx = int((msg_time - first_msg).days / quarter_length)
            if q_idx >= 4:
                q_idx = 3
            
            quarters[q_idx]['users'][msg['sender']]['messages'] += 1
            quarters[q_idx]['users'][msg['sender']]['chars'] += len(msg['text'])
        
        return quarters
    
    def analyze_user_lifecycle(self):
        """Analyze when users joined, left, or changed activity"""
        user_lifecycle = {}
        
        for user in self.users:
            user_msgs = [m for m in self.messages if m['sender'] == user]
            
            first_msg = user_msgs[0]['timestamp']
            last_msg = user_msgs[-1]['timestamp']
            msg_count = len(user_msgs)
            
            user_lifecycle[user] = {
                'first_message': first_msg,
                'last_message': last_msg,
                'days_active': (last_msg - first_msg).days,
                'message_count': msg_count,
                'avg_messages_per_active_day': msg_count / max(1, (last_msg - first_msg).days)
            }
        
        return user_lifecycle
    
    def analyze_activity_decline(self):
        """Identify users with significant activity decline"""
        quarters = self.analyze_intertemporal_patterns()
        
        user_trend = defaultdict(list)
        
        for q in quarters:
            for user, stats in q['users'].items():
                user_trend[user].append(stats['messages'])
        
        decline_analysis = {}
        for user, trend in user_trend.items():
            if len(trend) >= 2:
                # Calculate decline ratio (Q4 vs Q1)
                q1_msgs = trend[0] if trend[0] > 0 else 1
                q4_msgs = trend[-1] if trend[-1] > 0 else 0
                decline_ratio = (q1_msgs - q4_msgs) / q1_msgs * 100
                
                # Detect different patterns
                if trend[-1] == 0 and trend[-2] == 0:
                    status = "Inactive"
                elif decline_ratio > 70:
                    status = "Significant Decline"
                elif decline_ratio > 30:
                    status = "Moderate Decline"
                elif q4_msgs > q1_msgs * 1.5:
                    status = "Increased Activity"
                else:
                    status = "Stable"
                
                decline_analysis[user] = {
                    'status': status,
                    'decline_ratio': decline_ratio,
                    'q1_messages': trend[0],
                    'q4_messages': trend[-1],
                    'trend': trend
                }
        
        return decline_analysis
    
    def create_visualizations(self, output_dir="visualizations"):
        """Create comprehensive visualization suite"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        
        # 1. User Activity Distribution (Bar + Pie)
        self._viz_user_activity(output_dir)
        
        # 2. Message Composition (Pie chart)
        self._viz_message_composition(output_dir)
        
        # 3. Temporal Heatmap (Hour x Weekday)
        self._viz_temporal_heatmap(output_dir)
        
        # 4. Engagement Trends Over Time
        self._viz_engagement_trends(output_dir)
        
        # 5. Response Time Distribution
        self._viz_response_times(output_dir)
        
        # 6. User Burstiness Analysis
        self._viz_burstiness(output_dir)
        
        # 7. Sentiment Scores
        self._viz_sentiment(output_dir)
        
        # 8. Network Graph
        self._viz_network(output_dir)
        
        # 8a. Network Heatmap (Alternative visualization)
        self._viz_network_heatmap(output_dir)
        
        # 9. Time Gap Distribution
        self._viz_time_gaps(output_dir)
        
        # 10. Language Patterns
        self._viz_language(output_dir)
        
        # 11. Message Length Distribution
        self._viz_message_lengths(output_dir)
        
        # 12. Peak Hours Analysis
        self._viz_peak_hours(output_dir)
        
        # 13. Word Frequency
        self._viz_word_frequency(output_dir)
        
        # 14. User Dominance
        self._viz_user_dominance(output_dir)
        
        # 15. Conversation Participation
        self._viz_conversation_participation(output_dir)
        
        # 16. Inter-temporal Analysis
        self._viz_intertemporal_patterns(output_dir)
        
        # 17. User Lifecycle
        self._viz_user_lifecycle(output_dir)
        
        # 18. Activity Decline Analysis
        self._viz_activity_decline(output_dir)
        
        # 19. Network Data Export
        self._export_network_data(output_dir)
        
        print(f"\n✓ All visualizations generated in {output_dir}/")
    
    def _viz_user_activity(self, output_dir):
        """User activity bar chart and pie chart"""
        user_stats = self.analyze_user_stats()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top 15 users
        top_users = user_stats[:15]
        users = [u[0] for u in top_users]
        messages = [u[1]['message_count'] for u in top_users]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(users)))
        ax1.barh(users, messages, color=colors)
        ax1.set_xlabel('Message Count', fontsize=12)
        ax1.set_title('Top 15 Most Active Users', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # Pie chart
        labels = [u[0] for u in top_users[:8]]
        sizes = [u[1]['message_count'] for u in top_users[:8]]
        other_sum = sum([u[1]['message_count'] for u in user_stats[8:]])
        labels.append('Others')
        sizes.append(other_sum)
        
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Message Distribution by User', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_user_activity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_message_composition(self, output_dir):
        """Message type composition"""
        msg_stats = self.analyze_messages()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        types = ['Text', 'Images', 'Stickers', 'Links']
        counts = [
            msg_stats['text_messages'],
            msg_stats['image_messages'],
            msg_stats['sticker_messages'],
            msg_stats['link_messages']
        ]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        wedges, texts, autotexts = ax.pie(counts, labels=types, autopct='%1.1f%%', 
                                            colors=colors, startangle=90, textprops={'fontsize': 12})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Message Type Distribution', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_message_composition.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_temporal_heatmap(self, output_dir):
        """Hourly x Weekday heatmap"""
        heatmap_data = np.zeros((24, 7))
        
        for msg in self.messages:
            hour = msg['hour']
            weekday = msg['weekday']
            heatmap_data[hour][weekday] += 1
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Message Count'})
        
        hours = [f'{h:02d}:00' for h in range(24)]
        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        ax.set_xticklabels(weekdays)
        ax.set_yticklabels(hours, rotation=0)
        ax.set_xlabel('Day of Week', fontsize=12)
        ax.set_ylabel('Hour of Day', fontsize=12)
        ax.set_title('Activity Heatmap: Hour × Weekday', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/03_temporal_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_engagement_trends(self, output_dir):
        """Engagement trends over time"""
        trends = self.analyze_engagement_trends()
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Messages per day
        ax = axes[0]
        dates = trends['dates']
        # Filter out dates before 2022 (remove any 1970 artifacts)
        valid_indices = [i for i, d in enumerate(dates) if d.year >= 2022]
        dates = [dates[i] for i in valid_indices]
        messages_per_day = [trends['messages_per_day'][i] for i in valid_indices]
        users_per_day = [trends['users_per_day'][i] for i in valid_indices]
        
        ax.plot(dates, messages_per_day, marker='o', linestyle='-', 
                linewidth=1, markersize=3, color='#3498db', label='Messages/Day')
        
        # Add trend line
        if trends['trend_messages'] is not None:
            z = np.poly1d(trends['trend_messages'])
            ax.plot(dates, z(range(len(dates))), "r--", linewidth=2, label='Trend (3rd degree)')
        
        ax.set_ylabel('Messages', fontsize=12)
        ax.set_title('Daily Message Volume Trend', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Users per day - FIX: use dates instead of range() to avoid ghost lines
        ax = axes[1]
        ax.fill_between(dates, users_per_day, 
                         alpha=0.3, color='#2ecc71')
        ax.plot(dates, users_per_day, marker='s', linestyle='-', 
                linewidth=2, markersize=4, color='#27ae60')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Unique Users', fontsize=12)
        ax.set_title('Daily Unique Participants', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_engagement_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_response_times(self, output_dir):
        """Response time analysis"""
        response_stats = self.analyze_response_times_in_conversation()
        
        if not response_stats:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Response time distribution - ALL users
        by_user = response_stats['by_user']
        users = sorted(by_user.keys(), key=lambda x: by_user[x]['avg_response_time'])
        avg_times = [by_user[u]['avg_response_time'] for u in users]
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(users)))
        ax1.barh(users, avg_times, color=colors)
        ax1.set_xlabel('Avg Response Time (minutes)', fontsize=12)
        ax1.set_title(f'All {len(users)} Users: Response Times (sorted)', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        
        # Distribution histogram
        all_times = []
        for user in response_stats['by_user']:
            all_times.extend([response_stats['by_user'][user]['avg_response_time']] * 
                            response_stats['by_user'][user]['responses'])
        
        ax2.hist(all_times, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        ax2.axvline(np.median(all_times), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(all_times):.1f} min')
        ax2.set_xlabel('Response Time (minutes)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Response Time Distribution (all instances)', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/05_response_times.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_burstiness(self, output_dir):
        """User burstiness analysis"""
        burst_stats = self.analyze_user_burstiness()
        
        # Filter users with bursts - show ALL users
        burst_data = [(u, burst_stats[u]['burst_frequency']) for u in burst_stats 
                      if burst_stats[u]['burst_count'] > 0]
        burst_data = sorted(burst_data, key=lambda x: x[1], reverse=True)
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(burst_data) * 0.25)))
        
        users = [u[0] for u in burst_data]
        frequencies = [u[1] for u in burst_data]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(users)))
        bars = ax.barh(users, frequencies, color=colors)
        
        ax.set_xlabel('Burst Frequency (%)', fontsize=12)
        ax.set_title(f'Message Burstiness: All {len(users)} Active Users\n(% of messages that are part of back-to-back sequences)', 
                     fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                   ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/06_burstiness.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_sentiment(self, output_dir):
        """Sentiment analysis via emoji usage"""
        sentiment = self.analyze_emoji_sentiment()
        
        users = list(sentiment.keys())
        scores = [sentiment[u]['sentiment_score'] for u in users]
        emoji_counts = [sentiment[u]['emoji_count'] for u in users]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, len(users) * 0.2)))
        
        # Sentiment scores - ALL users
        sorted_idx = np.argsort(scores)
        sorted_users = [users[i] for i in sorted_idx]
        sorted_scores = [scores[i] for i in sorted_idx]
        
        colors = ['#e74c3c' if s < 0 else '#2ecc71' if s > 0 else '#95a5a6' for s in sorted_scores]
        ax1.barh(sorted_users, sorted_scores, color=colors)
        ax1.set_xlabel('Sentiment Score (via emoji)', fontsize=12)
        ax1.set_title(f'All {len(users)} Users: Sentiment Ranking', fontsize=13, fontweight='bold')
        ax1.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Emoji usage vs sentiment
        ax2.scatter(emoji_counts, scores, s=100, alpha=0.6, color='#3498db')
        ax2.set_xlabel('Total Emoji Count', fontsize=12)
        ax2.set_ylabel('Sentiment Score', fontsize=12)
        ax2.set_title('Emoji Usage vs Sentiment Score', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line if enough data
        if len(emoji_counts) > 2:
            z = np.polyfit([e for e in emoji_counts if e > 0], [scores[i] for i, e in enumerate(emoji_counts) if e > 0], 1)
            p = np.poly1d(z)
            x_trend = sorted([e for e in emoji_counts if e > 0])
            if x_trend:
                ax2.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, label='Trend')
                ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/07_sentiment.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_network(self, output_dir):
        """Interaction network visualization"""
        network = self.analyze_user_networks()
        G = network['graph']
        
        if len(G) < 3:
            return
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Node sizes based on message count
        node_sizes = [len([m for m in self.messages if m['sender'] == node]) * 3 
                      for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#3498db', 
                               alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        nx.draw_networkx_edges(G, pos, width=[w/max_weight*3 for w in weights], 
                               alpha=0.4, edge_color='gray', ax=ax, 
                               connectionstyle='arc3,rad=0.1', arrows=True, 
                               arrowsize=10, arrowstyle='->')
        
        ax.set_title('User Interaction Network\n(Node size = activity, Edge width = interaction frequency)', 
                    fontsize=13, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/08_network_graph.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_time_gaps(self, output_dir):
        """Time gap distribution analysis"""
        gap_data = self.analyze_time_gaps()
        gaps = gap_data['gaps']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Linear scale - all data
        ax = axes[0, 0]
        ax.hist(gaps, bins=100, color='#3498db', edgecolor='black', alpha=0.7)
        ax.axvline(gap_data['threshold'], color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {gap_data["threshold"]:.0f} min')
        ax.set_xlabel('Time Gap (minutes)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Time Gap Distribution (All Data, Linear Scale)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Linear scale - filtered (up to 95th percentile)
        ax = axes[0, 1]
        filtered_gaps = [g for g in gaps if g <= gap_data['percentile_95']]
        ax.hist(filtered_gaps, bins=100, color='#2ecc71', edgecolor='black', alpha=0.7)
        ax.axvline(gap_data['percentile_50'], color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {gap_data["percentile_50"]:.1f} min')
        ax.axvline(gap_data['percentile_95'], color='orange', linestyle='--', linewidth=2, 
                   label=f'95th: {gap_data["percentile_95"]:.1f} min')
        ax.set_xlabel('Time Gap (minutes)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Time Gap Distribution (≤95th Percentile)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Log scale
        ax = axes[1, 0]
        ax.hist(gaps, bins=100, color='#e74c3c', edgecolor='black', alpha=0.7)
        ax.axvline(gap_data['threshold'], color='red', linestyle='--', linewidth=2, 
                   label=f'Conversation Threshold')
        ax.set_xscale('log')
        ax.set_xlabel('Time Gap (minutes, log scale)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Time Gap Distribution (Log Scale)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        # Statistics box
        ax = axes[1, 1]
        ax.axis('off')
        stats_text = f"""
TIME GAP STATISTICS

Mean:                        {gap_data['mean_gap']:.2f} min
Std Dev:                     {gap_data['std_dev']:.2f} min
Median:                      {gap_data['percentile_50']:.2f} min
25th Percentile:         {gap_data['percentile_25']:.2f} min
75th Percentile:         {gap_data['percentile_75']:.2f} min
90th Percentile:         {gap_data['percentile_90']:.2f} min
95th Percentile:         {gap_data['percentile_95']:.2f} min

Threshold:                   {gap_data['threshold']:.2f} min
(Conversations separated by gaps > {gap_data['threshold']/60:.1f} hours)

Min Gap:                     {min(gaps):.3f} min
Max Gap:                     {max(gaps):.1f} min
        """
        ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/09_time_gaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_language(self, output_dir):
        """Language pattern analysis"""
        lang_stats = self.analyze_language()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Language distribution
        languages = ['English Only', 'Chinese Only', 'Mixed', 'Emoji Messages']
        counts = [
            lang_stats['english_messages'],
            lang_stats['chinese_messages'],
            lang_stats['mixed_messages'],
            lang_stats['emoji_messages']
        ]
        
        colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        wedges, texts, autotexts = ax1.pie(counts, labels=languages, autopct='%1.1f%%', 
                                             colors=colors, startangle=90, textprops={'fontsize': 11})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax1.set_title('Language Pattern Distribution', fontsize=13, fontweight='bold')
        
        # Language by user (top users)
        user_langs = defaultdict(lambda: {'english': 0, 'chinese': 0, 'mixed': 0})
        
        for msg in self.messages:
            user = msg['sender']
            text = msg['text']
            has_english = bool(re.search(r'[a-zA-Z]', text))
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
            
            if has_english and has_chinese:
                user_langs[user]['mixed'] += 1
            elif has_english:
                user_langs[user]['english'] += 1
            elif has_chinese:
                user_langs[user]['chinese'] += 1
        
        # Top users
        top_users_list = self.analyze_user_stats()[:10]
        top_users = [u[0] for u in top_users_list]
        
        english_pct = []
        mixed_pct = []
        
        for user in top_users:
            total = user_langs[user]['english'] + user_langs[user]['chinese'] + user_langs[user]['mixed']
            if total > 0:
                english_pct.append(user_langs[user]['english'] / total * 100)
                mixed_pct.append(user_langs[user]['mixed'] / total * 100)
            else:
                english_pct.append(0)
                mixed_pct.append(0)
        
        x = np.arange(len(top_users))
        width = 0.35
        
        ax2.bar(x, english_pct, width, label='English Only', color='#3498db')
        ax2.bar(x, mixed_pct, width, bottom=english_pct, label='Mixed Language', color='#f39c12')
        
        ax2.set_ylabel('Percentage (%)', fontsize=12)
        ax2.set_title('Language Use by Top Users', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(top_users, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/10_language_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_message_lengths(self, output_dir):
        """Message length distribution analysis"""
        length_stats = self.analyze_message_length_distribution()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top left: Average message length by user
        ax = axes[0, 0]
        users = sorted(length_stats.keys(), key=lambda x: length_stats[x]['avg_length'], reverse=True)
        avg_lengths = [length_stats[u]['avg_length'] for u in users]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(users)))
        ax.barh(users, avg_lengths, color=colors)
        ax.set_xlabel('Average Message Length (characters)', fontsize=12)
        ax.set_title('Avg Message Length by User', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        
        # Top right: Message length distribution histogram
        ax = axes[0, 1]
        all_lengths = []
        for msg in self.messages:
            length = len(msg['text'])
            if length < 1000:  # Filter out extreme outliers for visualization
                all_lengths.append(length)
        
        ax.hist(all_lengths, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        ax.axvline(np.median(all_lengths), color='red', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(all_lengths):.0f}')
        ax.axvline(np.mean(all_lengths), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(all_lengths):.0f}')
        ax.set_xlabel('Message Length (characters)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Message Length Distribution (all messages < 1000 chars)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom left: Std deviation of message length
        ax = axes[1, 0]
        users_std = sorted(length_stats.keys(), key=lambda x: length_stats[x]['std_length'], reverse=True)
        std_lengths = [length_stats[u]['std_length'] for u in users_std]
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(users_std)))
        ax.barh(users_std, std_lengths, color=colors)
        ax.set_xlabel('Std Dev of Message Length', fontsize=12)
        ax.set_title('Message Length Variability by User', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        
        # Bottom right: Total chars per user
        ax = axes[1, 1]
        users_chars = sorted(length_stats.keys(), key=lambda x: length_stats[x]['total_chars'], reverse=True)
        total_chars = [length_stats[u]['total_chars'] for u in users_chars]
        
        colors = plt.cm.autumn(np.linspace(0, 1, len(users_chars)))
        ax.barh(users_chars, total_chars, color=colors)
        ax.set_xlabel('Total Characters Used', fontsize=12)
        ax.set_title('Total Character Output by User', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/11_message_lengths.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_peak_hours(self, output_dir):
        """Peak activity hours analysis"""
        peak_hours = self.analyze_peak_hours()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        hours = [p['hour'] for p in peak_hours]
        messages = [p['messages'] for p in peak_hours]
        users = [p['unique_users'] for p in peak_hours]
        avg_lengths = [p['avg_msg_length'] for p in peak_hours]
        
        # Messages per hour
        ax = axes[0, 0]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(hours)))
        ax.bar(hours, messages, color=colors, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Message Count', fontsize=12)
        ax.set_title('Messages by Hour of Day', fontsize=13, fontweight='bold')
        ax.set_xticks(hours)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Unique users per hour
        ax = axes[0, 1]
        ax.plot(hours, users, marker='o', linestyle='-', linewidth=2, markersize=8, color='#2ecc71')
        ax.fill_between(hours, users, alpha=0.3, color='#2ecc71')
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Unique Users', fontsize=12)
        ax.set_title('Participation by Hour', fontsize=13, fontweight='bold')
        ax.set_xticks(hours)
        ax.grid(True, alpha=0.3)
        
        # Average message length per hour
        ax = axes[1, 0]
        ax.plot(hours, avg_lengths, marker='s', linestyle='-', linewidth=2, markersize=6, color='#e74c3c')
        ax.fill_between(hours, avg_lengths, alpha=0.2, color='#e74c3c')
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Avg Message Length', fontsize=12)
        ax.set_title('Message Depth by Hour', fontsize=13, fontweight='bold')
        ax.set_xticks(hours)
        ax.grid(True, alpha=0.3)
        
        # Top 3 peak hours detail
        ax = axes[1, 1]
        ax.axis('off')
        top_3 = sorted(peak_hours, key=lambda x: x['messages'], reverse=True)[:3]
        stats_text = "TOP 3 PEAK HOURS\n\n"
        for i, peak in enumerate(top_3, 1):
            stats_text += f"{i}. {peak['hour']:02d}:00-{peak['hour']+1:02d}:00\n"
            stats_text += f"   Messages: {peak['messages']}\n"
            stats_text += f"   Users: {peak['unique_users']}\n"
            stats_text += f"   Avg Length: {peak['avg_msg_length']:.0f} chars\n\n"
        
        ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/12_peak_hours.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_word_frequency(self, output_dir):
        """Top words used in the group"""
        word_freq = self.analyze_word_frequency()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top 20 words
        words = list(word_freq.keys())[:20]
        counts = list(word_freq.values())[:20]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(words)))
        ax1.barh(words, counts, color=colors, edgecolor='black')
        ax1.set_xlabel('Frequency', fontsize=12)
        ax1.set_title('Top 20 Most Used Words', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        
        # Word frequency distribution (log scale)
        ax2.loglog(range(1, len(word_freq) + 1), sorted(word_freq.values(), reverse=True), 'o-', 
                  color='#3498db', linewidth=2, markersize=6, alpha=0.7)
        ax2.set_xlabel('Word Rank', fontsize=12)
        ax2.set_ylabel('Frequency (log scale)', fontsize=12)
        ax2.set_title('Word Frequency Distribution (Zipfian)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/13_word_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_user_dominance(self, output_dir):
        """User dominance analysis"""
        dominance = self.analyze_user_dominance()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(dominance) * 0.2)))
        
        # Message share
        ax = axes[0]
        users = sorted(dominance.keys(), key=lambda x: dominance[x]['message_share'], reverse=True)
        msg_shares = [dominance[u]['message_share'] for u in users]
        
        colors = plt.cm.Spectral(np.linspace(0, 1, len(users)))
        ax.barh(users, msg_shares, color=colors)
        ax.set_xlabel('Message Share (%)', fontsize=12)
        ax.set_title('Message Volume Dominance', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Character share
        ax = axes[1]
        users_char = sorted(dominance.keys(), key=lambda x: dominance[x]['character_share'], reverse=True)
        char_shares = [dominance[u]['character_share'] for u in users_char]
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(users_char)))
        ax.barh(users_char, char_shares, color=colors)
        ax.set_xlabel('Character Share (%)', fontsize=12)
        ax.set_title('Content Volume Dominance', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/14_user_dominance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_conversation_participation(self, output_dir):
        """Conversation participation analysis"""
        participation = self.analyze_conversation_participation()
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(participation) * 0.2)))
        
        users = sorted(participation.keys(), key=lambda x: participation[x], reverse=True)
        counts = [participation[u] for u in users]
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(users)))
        bars = ax.barh(users, counts, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Conversations Participated In', fontsize=12)
        ax.set_title(f'Conversation Participation Rate (All {len(users)} Users)', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                   ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/15_conversation_participation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_intertemporal_patterns(self, output_dir):
        """Intertemporal analysis - how user behavior changes over quarters"""
        quarters = self.analyze_intertemporal_patterns()
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Get top 10 users
        top_users = [u[0] for u in self.analyze_user_stats()[:10]]
        
        # Messages per user per quarter
        ax = axes[0, 0]
        quarter_labels = [f"Q{q['quarter']}\n{q['start'].strftime('%y-%m')}" for q in quarters]
        
        for user in top_users:
            msgs_per_q = [q['users'][user]['messages'] for q in quarters]
            ax.plot(quarter_labels, msgs_per_q, marker='o', label=user, linewidth=2, markersize=6)
        
        ax.set_ylabel('Messages', fontsize=12)
        ax.set_title('Top 10 Users: Message Trends Over Time', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Character output per quarter
        ax = axes[0, 1]
        for user in top_users:
            chars_per_q = [q['users'][user]['chars'] for q in quarters]
            ax.plot(quarter_labels, chars_per_q, marker='s', label=user, linewidth=2, markersize=6)
        
        ax.set_ylabel('Total Characters', fontsize=12)
        ax.set_title('Top 10 Users: Content Volume Over Time', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Stacked bar: unique users per quarter
        ax = axes[1, 0]
        unique_per_q = [len(q['users']) for q in quarters]
        colors_q = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        ax.bar(quarter_labels, unique_per_q, color=colors_q, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Unique Users', fontsize=12)
        ax.set_title('Group Size Over Time (Unique Participants per Quarter)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(unique_per_q):
            ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Total messages per quarter
        ax = axes[1, 1]
        total_msgs_per_q = [sum(u['messages'] for u in q['users'].values()) for q in quarters]
        ax.bar(quarter_labels, total_msgs_per_q, color=colors_q, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Total Messages', fontsize=12)
        ax.set_title('Group Activity Over Time (Total Messages per Quarter)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(total_msgs_per_q):
            ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/16_intertemporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_user_lifecycle(self, output_dir):
        """User join/leave patterns"""
        lifecycle = self.analyze_user_lifecycle()
        
        fig, ax = plt.subplots(figsize=(16, max(8, len(lifecycle) * 0.2)))
        
        users = sorted(lifecycle.keys(), key=lambda x: lifecycle[x]['first_message'])
        first_msgs = [lifecycle[u]['first_message'] for u in users]
        last_msgs = [lifecycle[u]['last_message'] for u in users]
        active_days = [lifecycle[u]['days_active'] for u in users]
        msg_counts = [lifecycle[u]['message_count'] for u in users]
        
        # Create timeline
        y_pos = np.arange(len(users))
        colors = plt.cm.viridis(np.linspace(0, 1, len(users)))
        
        for i, user in enumerate(users):
            start = mdates.date2num(first_msgs[i])
            end = mdates.date2num(last_msgs[i])
            ax.barh(i, end - start, left=start, height=0.6, 
                   color=colors[i], edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(users, fontsize=9)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title('User Lifecycle Timeline\n(Bar shows duration from first to last message)', 
                    fontsize=13, fontweight='bold')
        
        # Format x-axis as dates
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/17_user_lifecycle.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_activity_decline(self, output_dir):
        """User activity decline analysis"""
        decline = self.analyze_activity_decline()
        
        # Categorize users
        categories = defaultdict(list)
        for user, data in decline.items():
            categories[data['status']].append((user, data))
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(decline) * 0.2)))
        
        # Sort by status then by decline ratio
        sorted_users = []
        status_order = ['Increased Activity', 'Stable', 'Moderate Decline', 'Significant Decline', 'Inactive']
        color_map = {'Increased Activity': '#2ecc71', 'Stable': '#3498db', 
                    'Moderate Decline': '#f39c12', 'Significant Decline': '#e74c3c', 'Inactive': '#95a5a6'}
        
        for status in status_order:
            if status in categories:
                sorted_by_decline = sorted(categories[status], key=lambda x: x[1]['decline_ratio'], reverse=True)
                sorted_users.extend(sorted_by_decline)
        
        users = [u[0] for u in sorted_users]
        declines = [u[1]['decline_ratio'] for u in sorted_users]
        statuses = [u[1]['status'] for u in sorted_users]
        colors = [color_map[s] for s in statuses]
        
        bars = ax.barh(users, declines, color=colors, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Activity Change from Q1 to Q4 (%)', fontsize=12)
        ax.set_title(f'User Activity Decline Analysis (All {len(users)} Users)', fontsize=13, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax.invert_yaxis()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[status], label=status, edgecolor='black') 
                          for status in status_order if status in categories]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/18_activity_decline.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _export_network_data(self, output_dir):
        """Export network data to markdown for high-quality visualization"""
        network = self.analyze_user_networks()
        G = network['graph']
        
        # Calculate centrality metrics
        out_degree = dict(G.out_degree(weight='weight'))
        in_degree = dict(G.in_degree(weight='weight'))
        betweenness = nx.betweenness_centrality(G)
        
        markdown = "# User Interaction Network Data\n\n"
        markdown += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += f"## Network Statistics\n"
        markdown += f"- **Nodes (Users):** {len(G)}\n"
        markdown += f"- **Edges (Interactions):** {len(G.edges())}\n"
        markdown += f"- **Network Density:** {network['density']:.4f}\n"
        markdown += f"- **Average Clustering Coefficient:** {network['avg_clustering']:.4f}\n\n"
        
        # Top influencers by outgoing interactions
        markdown += "## Top Influencers (Most Interactions Initiated)\n"
        markdown += "| User | Outgoing Connections | Total Interactions |\n"
        markdown += "|------|---------------------|-------------------|\n"
        
        top_out = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:15]
        for user, score in top_out:
            markdown += f"| {user} | {score:.1f} | {out_degree[user] + in_degree[user]:.1f} |\n"
        
        # All network edges
        markdown += "\n## All User Interactions\n"
        markdown += "| From User | To User | Interaction Count |\n"
        markdown += "|-----------|---------|------------------|\n"
        
        edges_sorted = sorted(network['interaction_pairs'].items(), key=lambda x: x[1], reverse=True)
        for (from_user, to_user), weight in edges_sorted:
            markdown += f"| {from_user} | {to_user} | {weight} |\n"
        
        with open(f'{output_dir}/network_data.md', 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        # Also create high-res network image
        fig, ax = plt.subplots(figsize=(24, 20))
        
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        
        node_sizes = [len([m for m in self.messages if m['sender'] == node]) * 5 
                      for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#3498db', 
                               alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        nx.draw_networkx_edges(G, pos, width=[w/max_weight*4 for w in weights], 
                               alpha=0.5, edge_color='gray', ax=ax, 
                               connectionstyle='arc3,rad=0.1', arrows=True, 
                               arrowsize=15, arrowstyle='->')
        
        ax.set_title('User Interaction Network (High Resolution)\nNode size = activity, Edge width = interaction frequency', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/19_network_highres.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _viz_network_heatmap(self, output_dir):
        """Create 2D heatmap visualization of user interactions (alternative to network graph)"""
        network = self.analyze_user_networks()
        interaction_pairs = network['interaction_pairs']
        
        # Get all unique users
        all_users = sorted(set([user for pair in interaction_pairs.keys() for user in pair] + 
                              [msg['sender'] for msg in self.messages]))
        
        # Create interaction matrix
        matrix = np.zeros((len(all_users), len(all_users)))
        user_to_idx = {user: idx for idx, user in enumerate(all_users)}
        
        for (from_user, to_user), count in interaction_pairs.items():
            i = user_to_idx[from_user]
            j = user_to_idx[to_user]
            matrix[i, j] = count
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(all_users) * 0.3), max(10, len(all_users) * 0.3)))
        
        # Use log scale for better visibility
        matrix_display = np.log1p(matrix)
        
        im = ax.imshow(matrix_display, cmap='YlOrRd', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(all_users)))
        ax.set_yticks(range(len(all_users)))
        ax.set_xticklabels(all_users, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(all_users, fontsize=9)
        
        ax.set_xlabel('To (Responding User)', fontsize=11, fontweight='bold')
        ax.set_ylabel('From (Message Sender)', fontsize=11, fontweight='bold')
        ax.set_title(f'User Interaction Heatmap (log scale)\nDarker = More Interactions', 
                    fontsize=12, fontweight='bold', pad=15)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Interaction Count (log scale)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/08a_network_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python whatsapp_analysis.py <path/to/chat.txt>")
        print("Example: python whatsapp_analysis.py alfafa/alfafa_20260219.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Verify file exists
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    # Determine output paths based on input file location
    input_dir = os.path.dirname(input_file)
    if not input_dir:
        input_dir = "."
    
    filename_base = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(input_dir, f"{filename_base}.md")
    output_viz_dir = os.path.join(input_dir, "visualizations")
    
    try:
        analyzer = ChatAnalyzer(input_file)
        
        print(f"📊 Analyzing: {input_file}")
        print(f"📄 Report will be saved to: {output_file}")
        print(f"📊 Visualizations will be saved to: {output_viz_dir}/")
        print()
        
        print("📊 Generating analysis reports...")
        analyzer.generate_report(output_file)
        
        print("🎨 Creating visualizations...")
        analyzer.create_visualizations(output_dir=output_viz_dir)
        
        print("\n✅ Analysis complete!")
        print(f"📄 Markdown report: {output_file}")
        print(f"📊 Visualizations: {output_viz_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
