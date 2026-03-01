import gradio as gr
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import json
import random

# Load the abnormal users trajectories data
df = pd.read_csv("data/abnormal_users_trajectories.csv")

# Load friends trajectories data from GitHub (large file)
GITHUB_RAW_URL = "https://raw.githubusercontent.com/chuchen2017/datastorage/main/users_have_overlap_with_abnormal.csv"
friends_df = pd.read_csv(GITHUB_RAW_URL)

# Load covisit data
with open("data/agent_pair_covisits.json", "r") as f:
    covisit_data = json.load(f)

# Get unique agent IDs
agent_ids = sorted(df['agent'].unique().tolist())

def plot_trajectory(agent_id):
    if agent_id is None:
        # Return empty figure if no agent selected
        fig = go.Figure()
        fig.update_layout(
            map_style="open-street-map",
            map=dict(
                center=dict(lat=35.6, lon=139.6),
                zoom=10
            ),
            height=600,
            title="Select an agent to view trajectory"
        )
        return fig
    
    # Filter data for selected agent
    agent_df = df[df['agent'] == agent_id].copy()
    
    # Sort by start timestamp to get correct trajectory order
    agent_df['start_timestamp'] = pd.to_datetime(agent_df['start_timestamp'])
    agent_df['stop_timestamp'] = pd.to_datetime(agent_df['stop_timestamp'])
    agent_df = agent_df.sort_values('start_timestamp')
    
    # Calculate visit frequency for each location
    location_counts = agent_df.groupby(['latitude', 'longitude']).size().reset_index(name='visit_frequency')
    agent_df = agent_df.merge(location_counts, on=['latitude', 'longitude'], how='left')
    
    # Prepare hover text with temporal information and visit frequency
    hover_texts = []
    for idx, row in agent_df.iterrows():
        hover_text = (
            f"<b>Stay ID:</b> {row['stay_id']}<br>"
            f"<b>Start:</b> {row['start_timestamp']}<br>"
            f"<b>Stop:</b> {row['stop_timestamp']}<br>"
            f"<b>Location:</b> ({row['latitude']:.6f}, {row['longitude']:.6f})<br>"
            f"<b>Visit Frequency:</b> {row['visit_frequency']}<br>"
            f"<b>Points Count:</b> {row['points_count']}<br>"
            f"<b>Label:</b> {row['matched_label']}"
        )
        hover_texts.append(hover_text)
    
    # Create the trajectory map
    fig = go.Figure()
    
    # Add trajectory line
    fig.add_trace(go.Scattermap(
        lat=agent_df['latitude'].tolist(),
        lon=agent_df['longitude'].tolist(),
        mode='lines+markers',
        marker=dict(
            size=8,
            color=agent_df['visit_frequency'].tolist(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Visit Frequency",
                thickness=15,
                len=0.7
            )
        ),
        line=dict(width=2, color='blue'),
        text=hover_texts,
        hoverinfo="text",
        name='Trajectory'
    ))
    
    # Add start point marker
    fig.add_trace(go.Scattermap(
        lat=[agent_df.iloc[0]['latitude']],
        lon=[agent_df.iloc[0]['longitude']],
        mode='markers',
        marker=dict(
            size=15,
            color='green',
            symbol='marker'
        ),
        text=['Start'],
        hoverinfo="text",
        name='Start'
    ))
    
    # Add end point marker
    fig.add_trace(go.Scattermap(
        lat=[agent_df.iloc[-1]['latitude']],
        lon=[agent_df.iloc[-1]['longitude']],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='marker'
        ),
        text=['End'],
        hoverinfo="text",
        name='End'
    ))
    
    # Calculate center for the map
    center_lat = agent_df['latitude'].mean()
    center_lon = agent_df['longitude'].mean()
    
    # Update layout
    fig.update_layout(
        map_style="open-street-map",
        hovermode='closest',
        map=dict(
            center=dict(
                lat=center_lat,
                lon=center_lon
            ),
            zoom=12
        ),
        height=600,
        title=f"Trajectory of Agent {agent_id}",
        showlegend=True
    )
    
    return fig

def get_covisit_stats(agent_id):
    """Get covisit statistics for the selected agent"""
    if agent_id is None:
        return go.Figure()
    
    # Find all pairs involving this agent
    covisit_counts = {}
    
    for pair_key, covisits in covisit_data.items():
        agents = pair_key.split('_')
        if str(agent_id) in agents:
            # Get the friend's ID (the other agent in the pair)
            friend_id = agents[1] if agents[0] == str(agent_id) else agents[0]
            covisit_counts[friend_id] = len(covisits)
    
    if not covisit_counts:
        fig = go.Figure()
        fig.update_layout(
            title=f"No covisit data found for Agent {agent_id}",
            height=400
        )
        return fig
    
    # Create histogram of covisit counts
    from collections import Counter
    count_distribution = Counter(covisit_counts.values())
    
    # Sort by number of covisits
    sorted_counts = sorted(count_distribution.items())
    x_values = [f"{count}" for count, _ in sorted_counts]
    y_values = [num_friends for _, num_friends in sorted_counts]
    
    fig = go.Figure(data=[
        go.Bar(
            x=x_values,
            y=y_values,
            marker=dict(color='steelblue'),
            hovertemplate='<b>Covisits:</b> %{x}<br><b>Number of Friends:</b> %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f"Covisit Distribution for Agent {agent_id}",
        xaxis_title="Number of Covisits",
        yaxis_title="Number of Friends",
        height=400,
        showlegend=False
    )
    
    return fig

def plot_friend_trajectory(agent_id, click_data):
    """Plot both agent's and friend's trajectories with covisit locations marked"""
    if agent_id is None or click_data is None:
        fig = go.Figure()
        fig.update_layout(
            map_style="open-street-map",
            map=dict(
                center=dict(lat=35.6, lon=139.6),
                zoom=10
            ),
            height=600,
            title="Enter covisit count and click 'Show Friend Trajectory' button"
        )
        return fig
    
    # Parse the clicked data to get the number of covisits
    try:
        clicked_covisit_count = int(click_data['points'][0]['x'])
    except:
        fig = go.Figure()
        fig.update_layout(
            map_style="open-street-map",
            map=dict(
                center=dict(lat=35.6, lon=139.6),
                zoom=10
            ),
            height=600,
            title="Enter covisit count and click 'Show Friend Trajectory' button"
        )
        return fig
    
    # Find all friends with this covisit count and get their covisit locations
    friends_with_count = []
    for pair_key, covisits in covisit_data.items():
        agents = pair_key.split('_')
        if str(agent_id) in agents and len(covisits) == clicked_covisit_count:
            friend_id = int(agents[1] if agents[0] == str(agent_id) else agents[0])
            friends_with_count.append((friend_id, pair_key))
    
    if not friends_with_count:
        fig = go.Figure()
        fig.update_layout(
            map_style="open-street-map",
            map=dict(
                center=dict(lat=35.6, lon=139.6),
                zoom=10
            ),
            height=600,
            title=f"No friends found with {clicked_covisit_count} covisits"
        )
        return fig
    
    # Randomly select one friend
    selected_friend, pair_key = random.choice(friends_with_count)
    
    # Get covisit locations for this pair
    covisit_locations = covisit_data[pair_key]
    
    # Get agent's trajectory
    agent_df = df[df['agent'] == agent_id].copy()
    agent_df['start_timestamp'] = pd.to_datetime(agent_df['start_timestamp'])
    agent_df['stop_timestamp'] = pd.to_datetime(agent_df['stop_timestamp'])
    agent_df = agent_df.sort_values('start_timestamp')
    
    # Calculate visit frequency for agent
    location_counts = agent_df.groupby(['latitude', 'longitude']).size().reset_index(name='visit_frequency')
    agent_df = agent_df.merge(location_counts, on=['latitude', 'longitude'], how='left')
    
    # Get friend's trajectory from friends_df
    friend_trajectory = friends_df[friends_df['agent'] == selected_friend].copy()
    
    if friend_trajectory.empty:
        fig = go.Figure()
        fig.update_layout(
            map_style="open-street-map",
            map=dict(
                center=dict(lat=35.6, lon=139.6),
                zoom=10
            ),
            height=600,
            title=f"No trajectory data found for Friend {selected_friend}"
        )
        return fig
    
    # Sort by start timestamp
    friend_trajectory['start_timestamp'] = pd.to_datetime(friend_trajectory['start_timestamp'])
    friend_trajectory['stop_timestamp'] = pd.to_datetime(friend_trajectory['stop_timestamp'])
    friend_trajectory = friend_trajectory.sort_values('start_timestamp')
    
    # Calculate visit frequency for friend
    friend_location_counts = friend_trajectory.groupby(['latitude', 'longitude']).size().reset_index(name='visit_frequency')
    friend_trajectory = friend_trajectory.merge(friend_location_counts, on=['latitude', 'longitude'], how='left')
    
    # Prepare hover text for agent
    agent_hover_texts = []
    for idx, row in agent_df.iterrows():
        hover_text = (
            f"<b>Agent {agent_id}</b><br>"
            f"<b>Stay ID:</b> {row['stay_id']}<br>"
            f"<b>Start:</b> {row['start_timestamp']}<br>"
            f"<b>Stop:</b> {row['stop_timestamp']}<br>"
            f"<b>Location:</b> ({row['latitude']:.6f}, {row['longitude']:.6f})<br>"
            f"<b>Visit Frequency:</b> {row['visit_frequency']}<br>"
            f"<b>Label:</b> {row['matched_label']}"
        )
        agent_hover_texts.append(hover_text)
    
    # Prepare hover text for friend
    friend_hover_texts = []
    for idx, row in friend_trajectory.iterrows():
        hover_text = (
            f"<b>Friend {selected_friend}</b><br>"
            f"<b>Stay ID:</b> {row['stay_id']}<br>"
            f"<b>Start:</b> {row['start_timestamp']}<br>"
            f"<b>Stop:</b> {row['stop_timestamp']}<br>"
            f"<b>Location:</b> ({row['latitude']:.6f}, {row['longitude']:.6f})<br>"
            f"<b>Visit Frequency:</b> {row['visit_frequency']}<br>"
            f"<b>Label:</b> {row['matched_label']}"
        )
        friend_hover_texts.append(hover_text)
    
    # Create the trajectory map
    fig = go.Figure()
    
    # Add agent's trajectory line
    fig.add_trace(go.Scattermap(
        lat=agent_df['latitude'].tolist(),
        lon=agent_df['longitude'].tolist(),
        mode='lines+markers',
        marker=dict(
            size=6,
            color=agent_df['visit_frequency'].tolist(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Agent Visit Freq",
                thickness=15,
                len=0.35,
                y=0.85
            )
        ),
        line=dict(width=2, color='blue'),
        text=agent_hover_texts,
        hoverinfo="text",
        name=f'Agent {agent_id}'
    ))
    
    # Add friend's trajectory line
    fig.add_trace(go.Scattermap(
        lat=friend_trajectory['latitude'].tolist(),
        lon=friend_trajectory['longitude'].tolist(),
        mode='lines+markers',
        marker=dict(
            size=6,
            color=friend_trajectory['visit_frequency'].tolist(),
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(
                title="Friend Visit Freq",
                thickness=15,
                len=0.35,
                y=0.45
            )
        ),
        line=dict(width=2, color='orange'),
        text=friend_hover_texts,
        hoverinfo="text",
        name=f'Friend {selected_friend}'
    ))
    
    # Add covisit locations with star markers
    covisit_lats = []
    covisit_lons = []
    covisit_hover = []
    
    for covisit in covisit_locations:
        # covisit structure: [bin_id, timestamp, agent_label, friend_label, duration, agent_overlap, friend_overlap, [lat, lon]]
        lat, lon = covisit[7]
        timestamp = covisit[1]
        duration = covisit[4]
        agent_label = covisit[2]
        friend_label = covisit[3]
        
        covisit_lats.append(lat)
        covisit_lons.append(lon)
        hover_text = (
            f"<b>🌟 COVISIT LOCATION 🌟</b><br>"
            f"<b>Time:</b> {timestamp}<br>"
            f"<b>Duration:</b> {duration}s<br>"
            f"<b>Agent {agent_id} Label:</b> {agent_label}<br>"
            f"<b>Friend {selected_friend} Label:</b> {friend_label}<br>"
            f"<b>Location:</b> ({lat:.6f}, {lon:.6f})"
        )
        covisit_hover.append(hover_text)
    
    if covisit_lats:
        fig.add_trace(go.Scattermap(
            lat=covisit_lats,
            lon=covisit_lons,
            mode='markers',
            marker=dict(
                size=35,
                color='gold',
                symbol='star',
                opacity=0.95
            ),
            text=covisit_hover,
            hoverinfo="text",
            name='Covisit Locations'
        ))
    
    # Calculate center for the map (using both trajectories)
    all_lats = agent_df['latitude'].tolist() + friend_trajectory['latitude'].tolist()
    all_lons = agent_df['longitude'].tolist() + friend_trajectory['longitude'].tolist()
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # Update layout
    fig.update_layout(
        map_style="open-street-map",
        hovermode='closest',
        map=dict(
            center=dict(
                lat=center_lat,
                lon=center_lon
            ),
            zoom=11
        ),
        height=600,
        title=f"Agent {agent_id} & Friend {selected_friend} - {clicked_covisit_count} Covisits (⭐ = Meeting Locations)",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        )
    )
    
    return fig

# Create Gradio interface
with gr.Blocks(title="Abnormal Users Trajectory Viewer") as demo:
    gr.Markdown("# Abnormal Users Trajectory Viewer")
    gr.Markdown("Select an agent ID to visualize their spatial-temporal trajectory on OpenStreetMap")
    
    with gr.Row():
        with gr.Column(scale=1):
            agent_dropdown = gr.Dropdown(
                choices=agent_ids,
                label="Select Agent ID",
                value=agent_ids[0] if agent_ids else None,
                interactive=True
            )
            gr.Markdown(f"**Total Agents:** {len(agent_ids)}")
        
        with gr.Column(scale=3):
            trajectory_map = gr.Plot(label="Agent Trajectory Map")
    
    gr.Markdown("---")
    gr.Markdown("## Friend Analysis")
    gr.Markdown("The bar chart shows the distribution of covisits with friends. Enter the number of covisits and click 'Show Friend' to view a random friend's trajectory with that covisit count.")
    
    with gr.Row():
        with gr.Column(scale=2):
            covisit_chart = gr.Plot(label="Covisit Distribution")
            with gr.Row():
                covisit_input = gr.Number(label="Number of Covisits", value=1, precision=0)
                show_friend_btn = gr.Button("Show Friend Trajectory")
        
        with gr.Column(scale=3):
            friend_map = gr.Plot(label="Friend Trajectory Map")
    
    # Load initial maps
    demo.load(plot_trajectory, inputs=[agent_dropdown], outputs=[trajectory_map])
    demo.load(get_covisit_stats, inputs=[agent_dropdown], outputs=[covisit_chart])
    
    # Update maps when agent selection changes
    agent_dropdown.change(plot_trajectory, inputs=[agent_dropdown], outputs=[trajectory_map])
    agent_dropdown.change(get_covisit_stats, inputs=[agent_dropdown], outputs=[covisit_chart])
    
    # Update friend map when button is clicked
    def show_friend_for_covisit(agent_id, covisit_count):
        if covisit_count is None:
            return plot_friend_trajectory(agent_id, None)
        return plot_friend_trajectory(agent_id, {'points': [{'x': int(covisit_count)}]})
    
    show_friend_btn.click(show_friend_for_covisit, inputs=[agent_dropdown, covisit_input], outputs=[friend_map])

demo.launch()