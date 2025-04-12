import pandas as pd
import os
from datetime import datetime

LOG_FILE = "chat_logs.xlsx"

def log_to_excel(user_query, response, retrieved_context, relevance_score):
    log_data = {"Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "User Query": [user_query],
        "Response": [response],
        "Retrieved Context": [retrieved_context],
        "Relevance Score": [relevance_score]
    }

    df = pd.DataFrame(log_data)

    if not os.path.exists(LOG_FILE):
        df.to_excel(LOG_FILE, index=False)  # Create new file
    else:
        existing_df = pd.read_excel(LOG_FILE)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_excel(LOG_FILE, index=False)  # Append new data