import gradio as gr
import random
import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict

# File paths for storing data
DATA_DIR = "data"
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.json")
PAIRWISE_FILE = os.path.join(DATA_DIR, "pairwise_results.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Elo ratings and other statistics
modalities = {
    "Fruits": {},
    "Actresses": {},
    "Cars": {},
    # Add more modalities as needed
}

image_folders = {
    "Fruits": "images/fruits",
    "Actresses": "images/Actresses",
    "Cars": "images/Cars",
    # Add paths for other modalities
}

def load_data():
    global modalities, pairwise_results
    if os.path.exists(RATINGS_FILE):
        with open(RATINGS_FILE, 'r') as f:
            modalities = json.load(f)
    
    if os.path.exists(PAIRWISE_FILE):
        with open(PAIRWISE_FILE, 'r') as f:
            pairwise_results = json.load(f)
    else:
        pairwise_results = {
            modality: defaultdict(lambda: defaultdict(lambda: [0, 0])) for modality in modalities
        }

def save_data():
    with open(RATINGS_FILE, 'w') as f:
        json.dump(modalities, f)
    
    with open(PAIRWISE_FILE, 'w') as f:
        json.dump(pairwise_results, f)

# Load existing data or initialize if not present
load_data()

# Load images and initialize ratings for each modality
for modality, folder in image_folders.items():
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            item_name = os.path.splitext(filename)[0]
            if item_name not in modalities[modality]:
                modalities[modality][item_name] = {
                    "rating": 1400,  # Initial Elo rating
                    "image": os.path.join(folder, filename),
                    "plays": 0
                }

current_modality = list(modalities.keys())[0]  # Default to the first modality
current_items = random.sample(list(modalities[current_modality].keys()), 2)

def calculate_elo(rating1, rating2, k=32, win=1):
    """Calculate new Elo ratings"""
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
    new_rating1 = rating1 + k * (win - expected1)
    new_rating2 = rating2 + k * ((1 - win) - expected2)
    return new_rating1, new_rating2

def update_ratings(winner, loser):
    """Update ratings based on the winner"""
    items = modalities[current_modality]
    new_winner_rating, new_loser_rating = calculate_elo(
        items[winner]["rating"], 
        items[loser]["rating"]
    )
    items[winner]["rating"] = new_winner_rating
    items[loser]["rating"] = new_loser_rating
    items[winner]["plays"] += 1
    items[loser]["plays"] += 1
    pairwise_results[current_modality][winner][loser][0] += 1
    pairwise_results[current_modality][winner][loser][1] += 1
    pairwise_results[current_modality][loser][winner][1] += 1
    save_data()

def get_random_items():
    """Get two random items from the current modality"""
    return random.sample(list(modalities[current_modality].keys()), 2)

def choose_item(choice):
    """Handle item selection"""
    global current_items
    item1, item2 = current_items
    if choice == item1:
        update_ratings(item1, item2)
    else:
        update_ratings(item2, item1)
    
    # Get new random items
    current_items = get_random_items()
    return (
        modalities[current_modality][current_items[0]]["image"],
        modalities[current_modality][current_items[1]]["image"],
        current_items[0],
        current_items[1],
        get_ratings_table(),
        get_pairwise_heatmap()
    )

def get_ratings_table():
    """Generate a DataFrame of current ratings"""
    items = modalities[current_modality]
    data = [(name, info["rating"], info["plays"]) for name, info in items.items()]
    df = pd.DataFrame(data, columns=["Item", "Rating", "Plays"])
    df = df.sort_values("Rating", ascending=False).reset_index(drop=True)
    df.index += 1  # Start index from 1 instead of 0
    return df

def get_pairwise_heatmap():
    """Generate an interactive heatmap of pairwise win rates"""
    items = modalities[current_modality]
    item_names = sorted(items.keys())
    n = len(item_names)
    win_rates = np.zeros((n, n))
    
    for i, item1 in enumerate(item_names):
        for j, item2 in enumerate(item_names):
            if i != j:
                wins, total = pairwise_results[current_modality][item1][item2]
                win_rates[i, j] = wins / total if total > 0 else 0.5
    
    fig = go.Figure(data=go.Heatmap(
        z=win_rates,
        x=item_names,
        y=item_names,
        hoverongaps=False,
        colorscale='YlGnBu',
        text=np.round(win_rates, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title=f"Pairwise Win Rates - {current_modality}",
        xaxis_title="Opponent",
        yaxis_title="Item",
        width=800,
        height=800,
    )
    
    return fig

def change_modality(new_modality):
    global current_modality, current_items
    current_modality = new_modality
    current_items = get_random_items()
    return (
        modalities[current_modality][current_items[0]]["image"],
        modalities[current_modality][current_items[1]]["image"],
        current_items[0],
        current_items[1],
        get_ratings_table(),
        get_pairwise_heatmap()
    )

with gr.Blocks() as demo:
    gr.Markdown("# Elo Rating System")
    
    with gr.Row():
        with gr.Column(scale=1):
            modality_dropdown = gr.Dropdown(
                choices=list(modalities.keys()),
                value=current_modality,
                label="Select Modality"
            )
        
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("Compare Items"):
                    gr.Markdown("Click on the button of the item you prefer. The Elo ratings will be updated accordingly.")
                    
                    with gr.Row():
                        img1 = gr.Image(value=modalities[current_modality][current_items[0]]["image"], label="Item 1")
                        img2 = gr.Image(value=modalities[current_modality][current_items[1]]["image"], label="Item 2")
                    
                    with gr.Row():
                        btn1 = gr.Button(current_items[0])
                        btn2 = gr.Button(current_items[1])
                
                with gr.TabItem("Ratings Table"):
                    ratings_table = gr.Dataframe(value=get_ratings_table())
                
                with gr.TabItem("Pairwise Results"):
                    pairwise_heatmap = gr.Plot()
    
    modality_dropdown.change(
        change_modality,
        inputs=[modality_dropdown],
        outputs=[img1, img2, btn1, btn2, ratings_table, pairwise_heatmap]
    )
    
    btn1.click(
        lambda: choose_item(current_items[0]),
        outputs=[img1, img2, btn1, btn2, ratings_table, pairwise_heatmap]
    )
    btn2.click(
        lambda: choose_item(current_items[1]),
        outputs=[img1, img2, btn1, btn2, ratings_table, pairwise_heatmap]
    )

demo.launch()
