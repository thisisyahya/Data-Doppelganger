"""
=========================================================================================
FAST BITWISE PERSONALITY MATCHING ENGINE
=========================================================================================
This script calculates personality compatibility across a dataset instantly. 
Instead of using slow loops, string comparisons, or heavy Machine Learning algorithms, 
it converts users' survey answers into 15-bit integers and compares them using 
hardware-level Bitwise operations and Pandas Vectorized Math.

HOW IT WORKS (Step-by-Step):

PART 1: DATA INGESTION & HEADER CLEANUP
    We load the raw CSV file. Because the original headers contain long, messy survey 
    questions and special characters, we instantly overwrite them with clean column 
    names ("Q1", "Q2", etc.) so we can reference them easily in code.

PART 2: THE ANSWER KEY
    We define a dictionary `ones_key`. For each question, we map the exact string of 
    the FIRST option. If a user's answer matches this string, they get a "1" bit. 
    If they picked the other option, they get a "0" bit.

PART 3: VECTORIZED CLEANING & COMPARISON
    We strip invisible spaces from the data to prevent exact-match errors. Then, 
    instead of looping through rows, we use Pandas "Broadcasting" to compare the 
    entire dataset against our answer key simultaneously. This generates a 2D matrix 
    of True/False values.

PART 4: MATRIX DOT PRODUCT (THE SPEED TRICK)
    
    Converting booleans to strings ("11010...") and then to integers is very slow. 
    Instead, we treat the 15 questions as 15 bits (MSB to LSB). We create an array of 
    powers of 2 (16384, 8192 ... 2, 1). By calculating the "Dot Product" of our boolean 
    matrix and this array, we mathematically convert the True/False answers directly 
    into base-10 integers in a fraction of a millisecond using C-level libraries.

PART 5: BITWISE MATH & POPCOUNT
    
    We take a target user's integer and compare it to everyone else's integer using 
    the Bitwise XOR operator (^). 
    - XOR gives a '1' everywhere the answers are DIFFERENT.
    - We use `.bit_count()` (Hardware Popcount) to count those differences.
    - We subtract the differences from 15 (Total Questions) to get the exact number 
      of MATCHES (which is equivalent to a Bitwise XNOR).

PART 6: SORTING & OUTPUT
    We assign the match scores, sort the dataset from highest to lowest, filter out 
    the target user (so they don't just match with themselves), and print the results!
=========================================================================================
"""

import pandas as pd
import numpy as np
import time

# ==========================================
# PART 1: DATA INGESTION & HEADER CLEANUP
# ==========================================
df = pd.read_csv("./personas.csv")

# Overwrite the headers completely to bypass Unicode characters and long strings.
df.columns = [
    "Timestamp", "Username", "Gender", 
    "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"
]

# ==========================================
# PART 2: THE ANSWER KEY
# ==========================================
# Define the exact strings that represent a "1" (MSB to LSB)
ones_key = {
    "Q1": "Going out with a lively group of friends",
    "Q2": "Clear, step-by-step instructions and facts",
    "Q3": "The Logical Side (Logic dominates my emotions)",
    "Q4": "I plan it out and finish it well before the deadline",
    "Q5": "To offer empathy and shared experience",
    "Q6": "Choosing the path with the least amount of risk",
    "Q7": "A clear mental image of what it looked like",
    "Q8": "Direct Communication (Saying exactly what is on my mind)",
    "Q9": "I just know internally based on my own standards",
    "Q10": "To restore the emotional connection (Harmony)",
    "Q11": 'The "vibe," tone of voice, and expressions',
    "Q12": "Just having them listen and agree that it sucks (Validation)",
    "Q13": "I like being free to change my mind and do things at the last minute.",
    "Q14": "I feel it and my mood shifts",
    "Q15": "I need to talk it out with someone else to truly understand what I’m thinking."
}

# ⏱️ START TIMER (Measuring just the logic engine)
start_time = time.perf_counter()

# Convert dictionary to Series and extract just the column names
correct_answers = pd.Series(ones_key)
q_cols = list(ones_key.keys())

# ==========================================
# PART 3: VECTORIZED CLEANING & COMPARISON
# ==========================================
# Strip invisible whitespace from all answers to prevent match failures
df[q_cols] = df[q_cols].apply(lambda x: x.str.strip())

# Create Boolean Matrix (True if they picked Option 1, False if Option 2)
bool_matrix = df[q_cols] == correct_answers

# ==========================================
# PART 4: MATRIX DOT PRODUCT (THE SPEED TRICK)
# ==========================================
# Create an array of powers of 2 for 15 bits: [16384, 8192, 4096, ... 4, 2, 1]
powers_of_2 = np.array([2**i for i in range(14, -1, -1)])

# Matrix Dot Product: Multiplies booleans by powers of 2 and sums them instantly.
# This assigns a unique base-10 integer to every user based on their 15 answers.
df["binary_int"] = bool_matrix.dot(powers_of_2)

# ==========================================
# PART 5: BITWISE MATH & POPCOUNT
# ==========================================
# Extract the target user (In this case, Row 26)
target_int = df.iloc[26]["binary_int"]
target_name = df.iloc[26]["Username"]

print(f"Finding matches for: {target_name}...\n")

# XOR the target integer against EVERY row, count the diff bits, and subtract from 15
df["Match_Score"] = df["binary_int"].apply(lambda x: 15 - (x ^ target_int).bit_count())

# ==========================================
# PART 6: SORTING & OUTPUT
# ==========================================
# Sort the DataFrame from highest score to lowest (ignoring the target user)
best_matches_df = df[df["Username"] != target_name].sort_values(by="Match_Score", ascending=False)

# ⏱️ STOP TIMER
end_time = time.perf_counter()

# Calculate and print execution time
execution_time = end_time - start_time
print(f"Algorithm Execution Time: {execution_time:.6f} seconds\n")

print("--- TOP MATCHES ---")
# Print the top 60 matches (Username and their Score out of 15)
print(best_matches_df[["Username", "Match_Score"]].head(60))