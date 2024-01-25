from __future__ import division
import os
import re
import numpy as np
from datasets import load_dataset

# Misc.
import warnings

warnings.filterwarnings('ignore')

from txtai.pipeline import Textractor

# Create textractor model
# textractor = Textractor()
# textractor = Textractor(sentences=True)
textractor = Textractor(paragraphs=True)


def GetAvgParagraphLength():
    dataset = load_dataset("climatebert/climate_sentiment")
    lengths = []

    for item in dataset['train']:
        # structure of item: {'text': '...', 'label': 0}
        text = item['text']
        length = len(text.split())
        lengths.append(length)
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    first_quartile = np.percentile(lengths, 25)
    third_quartile = np.percentile(lengths, 75)
    return [mean_length, median_length, first_quartile, third_quartile]

def is_bullet(line):
    bullet_patterns = [
        r"^\d+\.",  # Matches numbers followed by a period (e.g., 1., 2., 3.)
        r"^[a-zA-Z]\.",  # Matches single letters followed by a period (e.g., a., b., c.)
        r"^(i{1,3}|iv|v|vi{0,3})\."  # Matches Roman numerals followed by a period (i., ii., iii., iv.)
    ]
    for pattern in bullet_patterns:
        if re.search(pattern, line.strip()):
            return True
    return False

def PDFprocesser(path):
    endings = (".", "!", "?", ";", ":")
    papers = []
    papers_list = []
    # Get the average length of the paragraph
    mean_para_length, median_para_length, first_para_quartile, third_para_quartile = GetAvgParagraphLength()
    # clear the previous output
    open("./extracted_files.txt", "w", encoding='utf-8').close()
    open("./invalid_files.txt", "w", encoding='utf-8').close()
    for filename in [x for x in os.listdir('./NDCs') if x.endswith('.pdf')]:
        print(f"INFO: Start reading: {filename}")
        lines = textractor(f"./NDCs/{filename}")
        line_lengths = []
        for line in lines:
            line_lengths.append(len(line))
        if line_lengths:
            line_avg_length = int(np.mean(line_lengths))
            line_length_low_bound = np.percentile(line_lengths, 25)
            line_length_high_bound = np.percentile(line_lengths, 95)
        else:
            line_avg_length = 0
            line_length_low_bound = 0
            line_length_high_bound = 0
        # print('-------------')
        # print(line_avg_length)
        # print(line_length_low_bound)
        # print(line_length_high_bound)
        # Join lines into paragraphs
        paragraphs, current = [], []
        current_paragraph = ""
        pre_line_char_cnt = line_avg_length
        for i in range(len(lines)):
            line = lines[i]
            # Remove leading spaces
            line = line.lstrip()
            # Skip lines containing non-ASCII characters
            line = re.sub(r'[^\x00-\x7F]+', '', line)
            # Skip lines containing URLs
            if re.search(r'https?://\S+|www\.\S+', line):
                continue
            # Remove catalogues
            if re.search(r'\.\.\.\.', line):
                continue
            # Make sure beginning of paragraph is complete
            if current_paragraph == "":
                if line and line[0].islower() and not is_bullet(line):
                    continue
            # Remove references
            if line.lower() == "references" or line.lower() == "referencias":
                break
            # print("It's line ", i, ". It's length is ", len(line))
            # print(line)
            if line.endswith(endings):
                if abs(pre_line_char_cnt - len(line)) > 10:
                    current_paragraph += " " + line
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = ""
                else:
                    cur_len = len(current_paragraph.split())
                    if cur_len > third_para_quartile:
                        current_paragraph += " " + line
                        paragraphs.append(current_paragraph.strip())
                        current_paragraph = ""
                    elif i + 1 < len(lines) and len(lines[i + 1]) < line_avg_length:
                        current_paragraph += " " + line
                        paragraphs.append(current_paragraph.strip())
                        current_paragraph = ""
                    else:
                        current_paragraph += " " + line
            else:  # line does not end with "."
                # titles, sub-titles, annotations, etc.
                if len(line) < min(80, line_avg_length) or \
                        len(line) > line_length_high_bound:
                    continue
                else:
                    current_paragraph += " " + line
                    pre_line_char_cnt = len(line)
        if current_paragraph and current_paragraph.endswith(endings):
            paragraphs.append(current_paragraph.strip())
        if len(paragraphs) > 10:
            papers.append(paragraphs)
            with open("./extracted_files.txt", "a", encoding='utf-8') as f:
                f.write(f"{filename}\n")
                papers_list.append(filename)
            with open(f"./output/{filename.replace('.pdf', '.txt')}", "w", encoding='utf-8') as f:
                for paragraph in paragraphs:
                    f.write(paragraph)
                    f.write("\n\n")
        else:
            with open("./invalid_files.txt", "a", encoding='utf-8') as f:
                f.write(f"{filename}\n")


    # for paper in papers:
    #     for paragraph in paper:
    #         print(paragraph)
    #         print()

    return papers


path = "./NDCs"
PDFprocesser(path)
