# WhatsApp Group Chat Analysis Suite

A comprehensive Python toolkit for analyzing WhatsApp group chat exports across multiple conversations.

## Features

### Main Analysis Script: `whatsapp_analysis.py`
- Parse WhatsApp text exports (.txt format)
- Generate 20 professional visualizations
- Create detailed statistical reports
- Network analysis and interaction mapping

**Usage:**
```bash
python whatsapp_analysis.py <path/to/chat.txt>
```

**Output:**
- `visualizations/` - 20 PNG charts (01-20)
- `<chatname>.md` - Statistical report in same folder as input

**Visualizations Generated (20 total):**
1. User activity distribution
2. Message composition types
3. Temporal heatmap
4. Engagement trends
5. Response time analysis
6. Burstiness patterns
7. Sentiment analysis
8. Network graph (spring layout)
9. Time gap distribution
10. Language patterns
11. Message lengths
12. Peak hours analysis
13. Word frequency
14. User dominance
15. Conversation participation
16. Inter-temporal patterns (quarterly)
17. User lifecycle timeline
18. Activity decline detection
19. Network graph (high-resolution, 300 DPI)
20. **Network heatmap (2D matrix alternative)**

---

### Word Analysis Script: `word_analyzer.py`
Analyze specific word usage patterns in group chats.

**Usage:**
```bash
python word_analyzer.py <path/to/chat.txt>
```

**How it works:**
1. Looks for `keywords.txt` in the same folder as the chat file
2. Keywords should be one per line
3. Automatically processes all keywords
4. Generates usage breakdown and visualizations
5. Outputs all results to `word_analysis/` folder

**Example keywords.txt:**
```
Amen
Bible
Love
Prayer
Thanks
```

**Output:**
- CSV report with word statistics
- Bar chart of word usage by user
- Time-series visualization
- Saved in `word_analysis/` folder

---

## Input Format

### WhatsApp Export (.txt)
Standard WhatsApp text export format:
```
[DD/M/YYYY, HH:MM:SS] Sender Name: Message content
[DD/M/YYYY, HH:MM:SS] Another User: Message text
```

### Keywords File (optional, for word_analyzer.py)
One keyword per line, any case (auto-capitalize during search):
```
word1
word2
Word3
```

---

## Folder Structure

Each group chat profile should have:
```
group_name/
├── chat_export.txt          # WhatsApp export
├── keywords.txt             # (optional, for word_analyzer)
├── group_name.md            # Generated report
├── visualizations/          # Generated charts (20 PNGs)
└── word_analysis/           # Generated word reports
```

---

## Requirements

Python 3.7+ with packages:
- matplotlib
- seaborn
- networkx
- scipy
- numpy

---

## Notes

- Both scripts output results to the same folder as input
- Visualizations are saved at 300 DPI for publication quality
- Scripts handle multi-line messages automatically
- Network density and clustering coefficients calculated
- All analysis is deterministic and reproducible
