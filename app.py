import gradio as gr
import random
import os
import json
import yaml
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import threading
import time

# File paths for storing data
DATA_DIR = "data"
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.json")
PAIRWISE_FILE = os.path.join(DATA_DIR, "pairwise_results.json")
ITEMS_CONFIG_FILE = "items_config.yaml"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Global lock for file access
file_lock = threading.Lock()

def load_items_config():
    with open(ITEMS_CONFIG_FILE, 'r') as f:
        return yaml.safe_load(f)

# Load items configuration
items_config = load_items_config()

def load_data():
    with file_lock:
        if os.path.exists(RATINGS_FILE):
            with open(RATINGS_FILE, 'r') as f:
                modalities = json.load(f)
        else:
            modalities = {modality: {} for modality in items_config.keys()}
        
        if os.path.exists(PAIRWISE_FILE):
            with open(PAIRWISE_FILE, 'r') as f:
                pairwise_results = json.load(f)
        else:
            pairwise_results = {modality: {} for modality in items_config.keys()}
    
    return modalities, pairwise_results

def save_data(modalities, pairwise_results):
    with file_lock:
        with open(RATINGS_FILE, 'w') as f:
            json.dump(modalities, f)
        
        with open(PAIRWISE_FILE, 'w') as f:
            json.dump(pairwise_results, f)

def initialize_new_items():
    modalities, pairwise_results = load_data()
    
    for modality, items in items_config.items():
        if modality not in modalities:
            modalities[modality] = {}
        if modality not in pairwise_results:
            pairwise_results[modality] = {}

        current_item_names = set(item['name'] for item in items)
        
        # Remove items not in config
        items_to_remove = set(modalities[modality].keys()) - current_item_names
        for item_name in items_to_remove:
            del modalities[modality][item_name]
            if item_name in pairwise_results[modality]:
                del pairwise_results[modality][item_name]
            for other_item in pairwise_results[modality]:
                if item_name in pairwise_results[modality][other_item]:
                    del pairwise_results[modality][other_item][item_name]

        for item in items:
            item_name = item['name']
            if item_name not in modalities[modality]:
                modalities[modality][item_name] = {
                    "rating": 1400,
                    "image": item['image'],
                    "plays": 0
                }
            
            if item_name not in pairwise_results[modality]:
                pairwise_results[modality][item_name] = {}
            
            for other_item in items:
                other_item_name = other_item['name']
                if other_item_name != item_name:
                    if other_item_name not in pairwise_results[modality][item_name]:
                        pairwise_results[modality][item_name][other_item_name] = [0, 0]

    save_data(modalities, pairwise_results)

# Initialize new items
initialize_new_items()

def calculate_elo(rating1, rating2, k=8, win=1):
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    expected2 = 1 / (1 + 10 ** ((rating1 - rating2) / 400))
    new_rating1 = rating1 + k * (win - expected1)
    new_rating2 = rating2 + k * ((1 - win) - expected2)
    return new_rating1, new_rating2

def update_ratings(winner, loser, modality):
    max_retries = 5
    for _ in range(max_retries):
        modalities, pairwise_results = load_data()
        
        winner_rating = modalities[modality][winner]["rating"]
        loser_rating = modalities[modality][loser]["rating"]
        
        new_winner_rating, new_loser_rating = calculate_elo(winner_rating, loser_rating)
        
        modalities[modality][winner]["rating"] = new_winner_rating
        modalities[modality][loser]["rating"] = new_loser_rating
        modalities[modality][winner]["plays"] += 1
        modalities[modality][loser]["plays"] += 1
        
        pairwise_results[modality][winner][loser][0] += 1
        pairwise_results[modality][winner][loser][1] += 1
        pairwise_results[modality][loser][winner][1] += 1
        
        try:
            save_data(modalities, pairwise_results)
            break
        except json.JSONDecodeError:
            time.sleep(0.1)  # Wait a bit before retrying
    else:
        raise Exception("Failed to update ratings after multiple attempts")

def get_random_items(modality):
    modalities, _ = load_data()
    return random.sample(list(modalities[modality].keys()), 2)

def get_ratings_table(modality):
    modalities, _ = load_data()
    items = modalities[modality]
    data = [(name, info["rating"], info["plays"]) for name, info in items.items()]
    df = pd.DataFrame(data, columns=["Item", "Rating", "Plays"])
    df = df.sort_values("Rating", ascending=False).reset_index(drop=True)
    df.index += 1
    return df

def get_pairwise_heatmap(modality):
    modalities, pairwise_results = load_data()
    items = modalities[modality]
    
    sorted_items = sorted(items.items(), key=lambda x: x[1]['rating'], reverse=True)
    item_names = [item[0] for item in sorted_items]
    
    n = len(item_names)
    win_rates = np.zeros((n, n))
    
    for i, item1 in enumerate(item_names):
        for j, item2 in enumerate(item_names):
            if i != j:
                wins, total = pairwise_results[modality][item1][item2]
                win_rates[i, j] = wins / total if total > 0 else 0.5
            else:
                win_rates[i, j] = 0.5

    fig = go.Figure(data=go.Heatmap(
        z=win_rates,
        x=item_names,
        y=item_names,
        hoverongaps=False,
        colorscale=px.colors.diverging.RdYlGn,
        zmin=0,
        zmax=1,
        text=np.round(win_rates, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        xaxis_title="Opponent",
        yaxis_title="Item",
        width=800,
        height=800,
    )
    
    y_labels = [item for item in item_names[::-1]]
    fig.update_yaxes(ticktext=y_labels, tickvals=item_names[::-1])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis = dict(side ="top"))
    fig.update_xaxes(tickangle=90)
    return fig

current_modality = list(items_config.keys())[0]
current_items = get_random_items(current_modality)

def choose_item(choice):
    global current_items
    item1, item2 = current_items
    if choice == item1:
        update_ratings(item1, item2, current_modality)
    else:
        update_ratings(item2, item1, current_modality)
    
    current_items = get_random_items(current_modality)
    modalities, _ = load_data()
    return (
        modalities[current_modality][current_items[0]]["image"],
        modalities[current_modality][current_items[1]]["image"],
        current_items[0],
        current_items[1],
        get_ratings_table(current_modality),
        get_pairwise_heatmap(current_modality)
    )

def change_modality(new_modality):
    global current_modality, current_items
    current_modality = new_modality
    current_items = get_random_items(current_modality)
    modalities, _ = load_data()
    return (
        modalities[current_modality][current_items[0]]["image"],
        modalities[current_modality][current_items[1]]["image"],
        current_items[0],
        current_items[1],
        get_ratings_table(current_modality),
        get_pairwise_heatmap(current_modality)
    )

with gr.Blocks() as demo:
    gr.Markdown("# Elo Rating System")
    
    with gr.Row():
        with gr.Column(scale=1):
            modality_dropdown = gr.Dropdown(
                choices=list(items_config.keys()),
                value=current_modality,
                label="Select Modality"
            )
        
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("Compare Items"):
                    gr.Markdown("Click on the button of the item you prefer. The Elo ratings will be updated accordingly.")
                    
                    with gr.Row():
                        modalities, _ = load_data()
                        img1 = gr.Image(value=modalities[current_modality][current_items[0]]["image"], label="Item 1", width=500, height=500)
                        img2 = gr.Image(value=modalities[current_modality][current_items[1]]["image"], label="Item 2", width=500, height=500)
                    
                    with gr.Row():
                        btn1 = gr.Button(current_items[0])
                        btn2 = gr.Button(current_items[1])
                
                with gr.TabItem("Ratings Table"):
                    ratings_table = gr.Dataframe(value=get_ratings_table(current_modality))
                
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

demo.launch(share=True)
