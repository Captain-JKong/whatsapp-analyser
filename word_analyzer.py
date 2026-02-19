#!/usr/bin/env python3
"""
Word Analysis Script - Analyze specific words in chat and their usage patterns
Reads keywords from keywords.txt file in the same folder as the chat export
"""

import re
import sys
import os
from datetime import datetime
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class WordAnalyzer:
    def __init__(self, input_file):
        self.input_file = input_file
        self.messages = []
        self.parse_messages()
    
    def parse_messages(self):
        """Parse messages from WhatsApp export file"""
        # Support both formats: [DD/M/YYYY, HH:MM:SS] and [DD/MM/YYYY, HH:MM:SS am/pm]
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
                    'text': text
                }
            else:
                if current_message and line.strip():
                    current_message['text'] += '\n' + line
        
        if current_message:
            self.messages.append(current_message)
    
    def get_top_words(self, n=20):
        """Get top N most used words"""
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                     'that', 'this', 'it', 'its', 'he', 'she', 'they', 'we', 'you', 'i',
                     'image', 'omitted', 'sticker', 'http', 'www', 'url', 'link'}
        
        word_counts = Counter()
        
        for msg in self.messages:
            if 'image omitted' not in msg['text'].lower() and 'sticker omitted' not in msg['text'].lower():
                words = re.findall(r'\b[a-z\u4e00-\u9fff]+\b', msg['text'].lower())
                for word in words:
                    if word not in stopwords and len(word) > 2:
                        word_counts[word] += 1
        
        return word_counts.most_common(n)
    
    def get_messages_with_word(self, word, case_sensitive=False):
        """Get all messages containing a specific word"""
        matching_messages = []
        
        pattern = f'\\b{re.escape(word)}\\b'
        flags = 0 if case_sensitive else re.IGNORECASE
        
        for msg in self.messages:
            if re.search(pattern, msg['text'], flags):
                matching_messages.append(msg)
        
        return matching_messages
    
    def analyze_word_usage_by_user(self, word):
        """Analyze who uses a word most frequently"""
        matching_messages = self.get_messages_with_word(word)
        
        user_counts = Counter()
        for msg in matching_messages:
            user_counts[msg['sender']] += 1
        
        return dict(user_counts)
    
    def create_word_report(self, word, output_file=None):
        """Create a comprehensive report for analyzing a word"""
        print(f"\n=== ANALYSIS FOR WORD: '{word}' ===\n")
        
        messages = self.get_messages_with_word(word)
        usage_by_user = self.analyze_word_usage_by_user(word)
        
        total_count = len(messages)
        print(f"Total occurrences: {total_count}")
        print(f"Unique users who used it: {len(usage_by_user)}\n")
        
        print("Usage by User:")
        print("-" * 50)
        for user in sorted(usage_by_user.keys(), key=lambda x: usage_by_user[x], reverse=True):
            count = usage_by_user[user]
            percentage = (count / total_count * 100)
            print(f"  {user:30s} : {count:4d} ({percentage:5.1f}%)")
        
        print(f"\n\nSample Messages (first 10):")
        print("=" * 100)
        for i, msg in enumerate(messages[:10], 1):
            print(f"\n[{i}] {msg['sender']} ({msg['timestamp']}):")
            print(f"    {msg['text'][:200]}")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart of user usage
        users = sorted(usage_by_user.keys(), key=lambda x: usage_by_user[x], reverse=True)
        counts = [usage_by_user[u] for u in users]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(users)))
        ax1.barh(users, counts, color=colors, edgecolor='black')
        ax1.set_xlabel('Number of Uses', fontsize=12)
        ax1.set_title(f'Who uses "{word}" most?\n(Total: {total_count} occurrences)', 
                     fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, (user, count) in enumerate(zip(users, counts)):
            ax1.text(count, i, f' {count}', va='center', fontsize=10, fontweight='bold')
        
        # Time series of word usage
        dates = defaultdict(int)
        for msg in messages:
            date = msg['timestamp'].date()
            dates[date] += 1
        
        sorted_dates = sorted(dates.keys())
        usage_over_time = [dates[d] for d in sorted_dates]
        
        ax2.plot(sorted_dates, usage_over_time, marker='o', linestyle='-', 
                linewidth=2, markersize=5, color='#3498db')
        ax2.fill_between(range(len(sorted_dates)), usage_over_time, alpha=0.3, color='#3498db')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Daily Occurrences', fontsize=12)
        ax2.set_title(f'"{word}" Usage Over Time', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to {output_file}")
        
        plt.show()
        
        return {
            'word': word,
            'total_count': total_count,
            'unique_users': len(usage_by_user),
            'usage_by_user': usage_by_user,
            'messages': messages
        }

# Main execution - Batch mode
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python word_analyzer.py <path/to/chat.txt>")
        print("Example: python word_analyzer.py cc_cannot/cc_cannot.txt")
        print("\nNote: keywords.txt should be in the same folder as the chat file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Verify chat file exists
    if not os.path.exists(input_file):
        print(f"Error: Chat file not found: {input_file}")
        sys.exit(1)
    
    # Look for keywords.txt in same directory
    input_dir = os.path.dirname(input_file)
    if not input_dir:
        input_dir = "."
    
    keywords_file = os.path.join(input_dir, "key_words.txt")
    
    # Check if keywords file exists
    if not os.path.exists(keywords_file):
        print(f"Error: Keywords file not found: {keywords_file}")
        print("Please create a key_words.txt file in the same folder as your chat export")
        sys.exit(1)
    
    # Read keywords
    try:
        with open(keywords_file, 'r', encoding='utf-8') as f:
            keywords = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading keywords file: {e}")
        sys.exit(1)
    
    if not keywords:
        print("Error: keywords.txt is empty")
        sys.exit(1)
    
    print(f"📊 Word Analysis Tool - Batch Mode")
    print("=" * 60)
    print(f"Chat file: {input_file}")
    print(f"Keywords file: {keywords_file}")
    print(f"Keywords to analyze: {len(keywords)}")
    print()
    
    # Create output directory
    output_dir = os.path.join(input_dir, "word_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        analyzer = WordAnalyzer(input_file)
        
        print(f"📄 Analyzing {len(analyzer.messages)} messages...\n")
        print("-" * 60)
        
        # Process each keyword in batch mode
        all_results = {}
        for keyword in keywords:
            # Auto-capitalize the keyword (case-insensitive search)
            display_keyword = keyword.strip()
            
            # Get messages containing the word (case-insensitive)
            messages = analyzer.get_messages_with_word(display_keyword, case_sensitive=False)
            
            if not messages:
                print(f"⚠️  '{display_keyword}': 0 occurrences")
                continue
            
            usage_by_user = analyzer.analyze_word_usage_by_user(display_keyword)
            total_count = len(messages)
            
            all_results[display_keyword] = {
                'total_count': total_count,
                'unique_users': len(usage_by_user),
                'usage_by_user': usage_by_user,
                'messages': messages
            }
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Bar chart of user usage
            users = sorted(usage_by_user.keys(), key=lambda x: usage_by_user[x], reverse=True)
            counts = [usage_by_user[u] for u in users]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(users)))
            ax1.barh(users, counts, color=colors, edgecolor='black')
            ax1.set_xlabel('Number of Uses', fontsize=12)
            ax1.set_title(f'Who uses "{display_keyword}" most?\\n(Total: {total_count} occurrences)', 
                         fontsize=13, fontweight='bold')
            ax1.invert_yaxis()
            
            # Add value labels
            for i, (user, count) in enumerate(zip(users, counts)):
                ax1.text(count, i, f' {count}', va='center', fontsize=10, fontweight='bold')
            
            # Time series of word usage
            dates = defaultdict(int)
            for msg in messages:
                date = msg['timestamp'].date()
                dates[date] += 1
            
            sorted_dates = sorted(dates.keys())
            usage_over_time = [dates[d] for d in sorted_dates]
            
            ax2.plot(sorted_dates, usage_over_time, marker='o', linestyle='-', 
                    linewidth=2, markersize=5, color='#3498db')
            ax2.fill_between(range(len(sorted_dates)), usage_over_time, alpha=0.3, color='#3498db')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Daily Occurrences', fontsize=12)
            ax2.set_title(f'"{display_keyword}" Usage Over Time', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save visualization
            output_file = os.path.join(output_dir, f"word_{display_keyword.lower()}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print summary
            top_3_users = sorted(usage_by_user.items(), key=lambda x: x[1], reverse=True)[:3]
            top_3_str = ", ".join([f"{user} ({count})" for user, count in top_3_users])
            print(f"✓ '{display_keyword:15s}': {total_count:4d} occurrences  |  Top: {top_3_str}")
        
        print("-" * 60)
        print(f"\n✅ Word analysis complete!")
        print(f"📊 Visualizations saved to: {output_dir}/")
        print(f"📈 Analyzed {len(all_results)} keywords")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
