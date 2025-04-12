#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
import pandas as pd


connection = sqlite3.connect('ReportData.db')

""
cursor = connection.cursor()


cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")


tables = cursor.fetchall()


for i, table in enumerate(tables):
    table_name = table[0]  
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, connection)
    

    globals()[f"df{i+1}"] = df

    print(f"Preview of {table_name}:")
    print(df.head())
    print("\n")


cursor.close()
connection.close()


# In[2]:


# # Filter the DataFrame for the specific TestID
# captured_data = df2.loc[df2['TestID'] == '1_17032025_161954', 'CapturedData']

# # Print the captured data; using .iloc[0] to get the first matching row
# print("Captured Data for TestID 1_17032025_161954:")
# print(captured_data.iloc[0])


# In[3]:


# Prompt the user to select a DataFrame
df_choice = input("Select the Node: ")

# Create a mapping of choices to the DataFrame variables
dataframes = {
    "1": df1,
    "2": df2,
    "3": df3,
    "4": df4
}

# Get the selected DataFrame based on user input
if df_choice in dataframes:
    selected_df = dataframes[df_choice]
else:
    print("Invalid DataFrame selection. Please enter 1, 2, 3, or 4.")
    exit()  # Exit the script if an invalid selection is made

# Prompt the user to enter a TestID
test_id = input("Enter TestID: ")

# Filter the selected DataFrame for the provided TestID
captured_data = selected_df.loc[selected_df['TestID'] == test_id, 'CapturedData']

# Check if any matching record exists and print the result accordingly
if not captured_data.empty:
    print(f"Captured Data for TestID {test_id}:")
    print(captured_data.iloc[0])
else:
    print(f"No captured data found for TestID {test_id} in DataFrame {df_choice}.")


# In[4]:


section = input("Enter the section name: ")
col_headers_input = input("Enter column headers separated by commas (e.g., hardness, thickness, diameter, width): ")
col_headers = [col.strip() for col in col_headers_input.split(',')]
row_header_col = input("Enter the row header column name: ")


# In[5]:


# Prompt the user for a comma-separated list of fields
user_input = input("Enter the list of fields separated by commas:\n")

# Split the string by commas and remove extra whitespace from each field
fields_list = [field.strip() for field in user_input.split(",") if field.strip()]

# Display the resulting list
print("You entered the following fields:")
print(fields_list)


# In[6]:


# import re

# # Get the captured data as a string
# raw_data = captured_data.iloc[0]


# In[7]:


import re

# 0) Start from your raw CapturedData cell
raw = captured_data.iloc[0]

# 1) Replace literal crlf/<CR><LF> with real newlines
clean = re.sub(r'<CR><LF>|crlf', '\n', raw, flags=re.IGNORECASE)

# 2) Remove <HT> tabs and any 'nul' markers
clean = re.sub(r'<HT>', '\t', clean)
clean = re.sub(r'\bnul\b', '', clean, flags=re.IGNORECASE)

# 3) Remove any other non‑printables but keep letters, digits, dot, colon, space, tab, newline
clean = re.sub(r'[^\w\.\:\n\t ]+', '', clean)

# 4) Collapse multiple blank lines into one
clean = re.sub(r'\n+', '\n', clean).strip()

# 5) (Optional) lowercase everything
cleaned_data = clean.lower()


# print("Cleaned (with newlines):\n", cleaned_data)


# In[8]:


import re
import pandas as pd

def normalize_token(token):
    # Remove spaces and dots for a simplified comparison.
    # You can customize this further as needed.
    return re.sub(r'\s+|\.', '', token.lower())

def extract_table_from_flat_text(text,
                                 section,
                                 col_headers,
                                 units=None,
                                 table_type='normal',
                                 row_header_col=None):
    # Normalize newlines
    text = re.sub(r'\r\n|\r', '\n', text)
    text_lower = text.lower()
    sec_lower  = section.lower()

    # Find the section start
    idx = text_lower.find(sec_lower)
    if idx == -1:
        print(f"Section '{section}' not found.")
        return None
    lines = text[idx:].strip().split('\n')

    # Prepare normalized version of expected headers
    normalized_expected = [normalize_token(h) for h in col_headers]
    header_idx = None

    # Locate header line with normalization
    for i, line in enumerate(lines):
        toks = line.strip().split()
        # Normalize each token
        normalized_toks = [normalize_token(t) for t in toks]

        if table_type == 'normal':
            # Check if all expected headers appear in the normalized token list
            if all(exp_header in normalized_toks for exp_header in normalized_expected):
                header_idx = i
                break
        else:  # for matrix type tables
            if normalized_toks == normalized_expected:
                header_idx = i
                break
            # If there's a row header plus the expected columns
            if (len(normalized_toks) >= len(normalized_expected) + 1 and
                normalized_toks[1:1+len(normalized_expected)] == normalized_expected):
                header_idx = i
                break

    if header_idx is None:
        print(f"Could not locate header line for section '{section}'.")
        return None

    # Parse the data rows after the header line
    data_lines = lines[header_idx+1:]
    number_re = re.compile(r'^[+-]?\d+(?:\.\d+)?$')
    rows = []

    for line in data_lines:
        tokens = line.strip().split()
        if not tokens:
            continue

        if table_type == 'normal':
            nums = [t for t in tokens if number_re.match(t)]
            if len(nums) == len(col_headers):
                rows.append(nums)
        else:  # For matrix table type
            j = next((i for i, t in enumerate(tokens) if number_re.match(t)), None)
            if j is None:
                continue
            nums = tokens[j:j+len(col_headers)]
            if len(nums) == len(col_headers) and all(number_re.match(x) for x in nums):
                row_label = " ".join(tokens[:j])
                rows.append([row_label] + nums)

    # Build DataFrame
    if table_type == 'normal':
        df = pd.DataFrame(rows, columns=col_headers)
        if units:
            df_units = pd.DataFrame([units], columns=col_headers)
            df = pd.concat([df_units, df], ignore_index=True)
        return df
    else:
        if not row_header_col:
            raise ValueError("`row_header_col` is required for matrix tables.")
        cols = [row_header_col] + col_headers
        return pd.DataFrame(rows, columns=cols).set_index(row_header_col)


# In[9]:


# Default to 'normal' table if row_header_col is not provided or is empty
if not row_header_col:
    table_type = 'normal'
    df_stats = extract_table_from_flat_text(
        text=cleaned_data,
        section=section,
        col_headers=col_headers,
        table_type=table_type
    )
else:
    table_type = 'matrix'
    df_stats = extract_table_from_flat_text(
        text=cleaned_data,
        section=section,
        col_headers=col_headers,
        table_type=table_type,
        row_header_col=row_header_col
    )

# Print and assign to df_main if not already defined
print(df_stats)
if df_stats is not None and ('df_main' not in locals() or df_main is None):
    df_main = df_stats.copy()
    print(df_main)


# ### Normal Table

# In[10]:


# # Example: extract the Test Results section as a normal table
# df_stats = extract_table_from_flat_text(
#     text=cleaned_data,
#     section=section,
#     col_headers=col_headers,
#     table_type='normal'
# )

# print(df_stats)
# if df_stats is not None and 'df_main' not in locals():
#     df_main = df_stats.copy()
#     print(df_main)


# ### Matric

# In[11]:


# df_stats = extract_table_from_flat_text(
#     text=cleaned_data,
#     section=section,
#     col_headers=col_headers,
#     table_type='matrix',
#     row_header_col=row_header_col
# )

# print(df_stats)
# if df_stats is not None and df_main is None:
#     df_main = df_stats.copy()
#     print(df_main)


# In[12]:


if df_stats is not None and 'df_main' not in locals():
    df_main = df_stats.copy()
    print(df_main)


# In[13]:


# # Call the function to extract the "Run Parameters" section as a table
# df_stats = extract_table_from_flat_text(
#     text=cleaned_data,
#     section="rpm history",
#     col_headers=[
# "interval" ,"rpm"
#     ]
# )

# # Print the extracted DataFrame
# print(df_stats)

# if df_stats is not None and not df_stats.empty:
#     df_main = df_stats.copy()
#     print(df_main)
# else:
#     print("df_stats is None or empty. Ignoring.")


# In[14]:


# df_main.head()


# ### Table without header

# In[15]:


import re
import pandas as pd

def extract_measurement_values(text, section, col_headers):
    try:
        # Normalize newlines
        text = re.sub(r'\r\n|\r', '\n', text)
        text_lower = text.lower()
        sec_lower = section.lower()

        # Find the start of the section
        idx = text_lower.find(sec_lower)
        if idx == -1:
            print(f"Section '{section}' not found.")
            return None
        lines = text[idx:].strip().split('\n')

        # Parse the data lines after the section header
        data_lines = lines[1:]  # Skip the first line which is the header
        rows = []

        for line in data_lines:
            line = line.strip()
            if line:
                # Extract the data based on column count
                match = re.match(r'(\d+)\s*(:\s*(\d+))?', line)
                if match:
                    # Extracting the first column as "Measurement Number"
                    row_data = [match.group(1)]
                    if match.group(3):  # If there is a value after ':'
                        row_data.append(int(match.group(3)))
                    else:
                        row_data.append(None)  # Placeholder for missing value
                    rows.append(row_data)

        # Create a DataFrame from the extracted rows and user-defined column names
        df = pd.DataFrame(rows, columns=col_headers)
        return df

    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        return None

# Define your custom column names here, they can be more than two columns
col_headers = col_headers

meas_section = section
df_stats = extract_measurement_values(cleaned_data, meas_section, col_headers)

print(df_stats)


# In[16]:


if df_stats is not None and 'df_main' not in locals():
    df_main = df_stats.copy()
    print(df_main)


# ### Method for nested table

# In[17]:


import re

# Get the captured data as a string
raw_data = captured_data.iloc[0]

# Step 1: Remove all HTML-like tags (e.g., <CR><LF>, <NUL>, etc.)
cleaned_data = re.sub(r'<[^>]+>', '', raw_data)

# Step 2: Replace all characters except letters, digits, spaces, dot (.), slash (/), and colon (:) with space
cleaned_data = re.sub(r'[^A-Za-z0-9\s./:]+', ' ', cleaned_data)

# Step 3: Replace multiple spaces with a single space
cleaned_data = re.sub(r'\s+', ' ', cleaned_data)

# Step 4: Strip leading/trailing spaces
cleaned_data = cleaned_data.strip()

# Step 5: Convert to lowercase
cleaned_data = cleaned_data.lower()

# Print the fully cleaned data
print("Fully Cleaned Captured Data:")
print(cleaned_data)
""


# In[18]:


import re
import pandas as pd

def extract_table_from_flat_text(text, section, headers, units=None):
    # Normalize text: lowercase and replace multiple spaces with one
    text = re.sub(r'\s+', ' ', text.lower())

    # Find the section
    section_start = text.find(section.lower())
    if section_start == -1:
        print(f"Section '{section}' not found.")
        return None

    # Get the part after the section header
    section_text = text[section_start:]

    # Adjust regex to capture the run parameters based on your data format
    # The pattern now expects 17 columns, including the interval, time, rpm, and 12 'blx' columns
    row_pattern = r'(\d+)\s+(\d{3}:\d{2})\s+(\d+\.\d+)\s+([\d\.]+(?:\s+[\d\.]+){13})'

    matches = re.findall(row_pattern, section_text)

    # If no matches were found, print a message
    if not matches:
        print(f"No data matched the pattern in section '{section}'.")
        return None

    # Prepare the data in rows based on the matched result
    rows = []
    for match in matches:
        # Combine the match with the other numeric columns into a full row
        row = list(match[:3]) + match[3].split()
        rows.append(row)

    # Ensure the number of columns matches the length of the data
    if len(headers) != len(rows[0]):
        print(f"Warning: The number of columns in the headers ({len(headers)}) does not match the extracted data ({len(rows[0])})")
        headers = headers[:len(rows[0])]  # Adjust headers to match the number of columns

    # Build the DataFrame from the rows
    df = pd.DataFrame(rows, columns=headers)

    # If a units row is provided, insert it as the first row of the DataFrame
    if units:
        unit_df = pd.DataFrame([units], columns=headers)
        df = pd.concat([unit_df, df], ignore_index=True)

    return df


# In[19]:


# Call the function to extract the "Run Parameters" section as a table
df_stats = extract_table_from_flat_text(
    text=cleaned_data,
    section=section,
    headers=col_headers
)

# Print the extracted DataFrame
print(df_stats) 


# In[20]:


if df_stats is not None and 'df_main' not in locals():
    df_main = df_stats.copy()
    print(df_main)


# In[21]:


# Call the function to extract the "Run Parameters" section as a table
df_run = extract_table_from_flat_text(
    text=cleaned_data,
    section=section,
    headers=col_headers
)

# Print the extracted DataFrame
print(df_stats) 


# In[22]:


if df_stats is not None and 'df_main' not in locals():
    df_main = df_stats.copy()
    print(df_main)


# ### Auto Extract Table

# In[23]:


import re
import pandas as pd

def auto_extract_table(text, col_headers):
    """
    Given a report as free-form text and a list of column headers,
    this function scans the text for table rows (assumed to use a pipe '|' delimiter)
    and extracts rows that start with a time pattern (HH:MM:SS) as the first column.
    Returns a DataFrame with the provided headers.
    """
    # Split into lines
    lines = text.splitlines()

    # Collect candidate table lines: those containing the pipe delimiter and not just dashed lines.
    candidate_lines = []
    for line in lines:
        # Remove leading/trailing spaces and ignore lines that are just a series of dashes.
        line_clean = line.strip()
        if "|" in line_clean and set(line_clean) != {"-"}:
            candidate_lines.append(line_clean)

    # Debugging: show candidate lines
    # print("Candidate table lines:")
    # for cl in candidate_lines:
    #     print(cl)
    
    # Among these candidate lines, we want to exclude header rows that are not data.
    # One common trait in our sample is that data rows start with a time stamp (e.g., "15:52:56")
    time_pattern = re.compile(r'^\d{2}:\d{2}:\d{2}')
    
    data_rows = []
    for line in candidate_lines:
        # Split on pipe delimiter and remove empty entries (from trailing pipes, etc.)
        tokens = [token.strip() for token in line.split('|') if token.strip() != '']
        # If the first token matches the time pattern, consider this a data row.
        if tokens and time_pattern.match(tokens[0]):
            data_rows.append(tokens)
    
    # If no rows were extracted with the time heuristic, as a fallback,
    # try filtering by matching the expected number of columns.
    if not data_rows:
        for line in candidate_lines:
            tokens = [token.strip() for token in line.split('|')]
            if len(tokens) == len(col_headers):
                data_rows.append(tokens)
    
    if not data_rows:
        print("No table data rows found in the provided report text.")
        return None

    # Optionally, print the extracted rows for debugging
    print("Extracted Data Rows:")
    for row in data_rows:
        print(row)

    # Create a DataFrame from the extracted rows,
    # using the user-supplied column headers.
    # Note: If the number of tokens in a row differs from the length of col_headers,
    # additional adjustments may be needed.
    df = pd.DataFrame(data_rows, columns=col_headers)
    return df

# # Filter the DataFrame for the specific TestID
# captured_data = df3.loc[df3['TestID'] == '3_17032025_162745', 'CapturedData']

# # Print the captured data; using .iloc[0] to get the first matching row
# print("Captured Data for TestID 3_17032025_162407:")
# print(captured_data.iloc[0])


import re

# Get the captured data as a string
raw_data = captured_data.iloc[0]
# User-supplied column headers for the table
col_headers = col_headers

# Extract the table as a DataFrame
df_stats = auto_extract_table(raw_data, col_headers)

print("\nExtracted DataFrame:")
print(df_stats)


# In[24]:


if df_stats is not None and 'df_main' not in locals():
    df_main = df_stats.copy()
    print(df_main)


# ### Field Extraction into table

# In[25]:


print(cleaned_data)


# In[26]:


cleaned_data = re.sub(r'[.:\\|/\d]', '', cleaned_data)

print(cleaned_data)


# In[27]:


df = df_main.copy()


# In[28]:
# Take input from the user for maximum number of rows in the table
max_rows = int(input("Enter the maximum number of rows the table should have: "))

# Display the entered value
print(f"The maximum number of rows the table can have is: {max_rows}")


if len(df) > max_rows:
    # Drop rows from the bottom
    df = df.head(max_rows)

import re

# Function to clean strings
def clean_text(s):
    if isinstance(s, str):
        return re.sub(r'[.:\\|/\d]', '', s)  # Added forward slash /
    return s

# Clean column headers
df_main.columns = [clean_text(col) for col in df_main.columns]

# Clean all cell values
df_main = df_main.applymap(clean_text)

# Print cleaned DataFrame
# print(df_main)


# In[29]:


import re

# Function to remove each occurrence of a word or number using regex with word boundaries.
def remove_occurrence(text, item):
    # Convert item to string first
    item_str = str(item)
    pattern = r'\b' + re.escape(item_str) + r'\b'
    return re.sub(pattern, '', text)

# Example cleaned_data (your original full text string)
# cleaned_data = "..." 

# Remove each DataFrame index value (e.g. 0, 1, "average", etc.)
for idx in df_main.index:
    cleaned_data = remove_occurrence(cleaned_data, idx)

# Remove each DataFrame column header (e.g. "hardness", "thickness", etc.)
for col in df_main.columns:
    cleaned_data = remove_occurrence(cleaned_data, col)

# Remove each DataFrame cell value
for col in df_main.columns:
    for val in df_main[col]:
        val_str = str(val).strip()
        cleaned_data = remove_occurrence(cleaned_data, val_str)

# Remove extra whitespace introduced by removals
cleaned_data_modified = re.sub(r'\s+', ' ', cleaned_data).strip()

# # Optional: print result
# print(cleaned_data_modified)


# In[30]:

#
# print("Modified cleaned_data:")
# print(cleaned_data_modified)


# In[31]:


import pandas as pd
import re


# In[32]:


# fields_list = [
#     "brand",
#     "model number",
#     "serial number",
#     "instrument id",
#     "company name",
#     "department",
#     "user details user",
#     "role",
#     "user group",
#     "product details product name",
#     "product descr.",
#     "tablet shape",
#     "product parameters mode",
#     "delay",
#     "method",
#     "speed",
#     "back off",
#     "product units hardness unit",
#     "hardness precision",
#     "ud factor",
#     "weight unit",
#     "weight precision",
#     "length unit",
#     "length precision",
#     "method details method name",
#     "hardness samples",
#     "weight samples",
#     "thickness samples",
#     "diameter samples",
#     "width samples",
#     "test details test id",
#     "press identif.",
#     "batch identif.",
#     "container number",
#     "test comment",
#     "ex. thickness",
#     "start date/time",
#     "end date/time",
#     "print time"
# ]


# In[33]:


# fields_list = ["kloudface details brand","model no","brand","model no","instrument id","serial no","fw version"]


# In[34]:


import re

# Escape the field names to handle any special regex characters
pattern_fields = "|".join(re.escape(field) for field in fields_list)
pattern = rf'(?i)\b({pattern_fields})\b\s*:?\s*(.*?)(?=\b(?:{pattern_fields})\b\s*:?\s*|$)'

matches = re.findall(pattern, cleaned_data_modified, re.DOTALL)

# Build a dictionary from matches; keys are lowercased for consistency.
extracted = {key.strip().lower(): value.strip() for key, value in matches}

# Make sure each pre-defined field is present in the result. Use None if missing.
result_mapping = {}
for field in fields_list:
    result_mapping[field] = extracted.get(field.lower(), None)


# In[35]:


import pandas as pd

# Overwrite (ignore) the old df_main completely
df_new = pd.DataFrame([result_mapping])

# print("New df_main:")
# print(df_new)


# In[36]:

#
# df_new.head()
#
#
# # In[37]:
#
#
# df.head()
#

# In[38]:


# For the primary data, orient 'records' returns a list of row dictionaries.
data_dict = df_new.to_dict(orient='records')
# For the summary statistics, orient 'index' creates a dictionary keyed by statistic type.
stats_dict = df.to_dict(orient='index')


# In[39]:


# Combine both dictionaries into one.
combined = {
    "data": data_dict,
    "stats": stats_dict
}


# In[40]:


print(combined)


# In[41]:
print("Succefully Saved the configuration!")

import json
# Write the combined data to a JSON file.
with open('combined_data.json', 'w') as json_file:
    json.dump(combined, json_file, indent=4)
print("Succefully Saved the configuration!")


# In[ ]:




