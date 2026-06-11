import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_biometric_dashboard():
    log_file = "data/vitals_log.csv"
    
    # 1. Check if data exists
    if not os.path.exists(log_file):
        print("[ERROR] No vitals log data found. Run the recognizer script first!")
        return
        
    # 2. Read the log file into a Pandas DataFrame
    df = pd.read_csv(log_file)
    
    if df.empty:
        print("[INFO] Log file is currently empty. Record some mood variations first.")
        return

    print(f"[SUCCESS] Loaded {len(df)} biometric tracking records.")

    # Set up a professional plotting window canvas with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("The Witch Light - Behavioral Analytics Dashboard", fontsize=16, fontweight='bold')

    # --- GRAPH 1: Pie Chart of Your Overall Emotion Distribution ---
    # Calculate the percentage frequency of each tracked emotional state
    emotion_counts = df['Emotion'].value_counts()
    
    # Map matching clean color themes to each biometric category
    color_map = {
        'Happy': '#2ecc71',        # Emerald Green
        'Sad': '#3498db',          # Deep Blue
        'Angry': '#e74c3c',         # Crimson Red
        'Surprised': '#1abc9c',     # Cyan Blue
        'Deep Focus': '#9b59b6',    # Amethyst Purple
        'Neutral State': '#f1c40f', # Amber Yellow
        'Calibrating... Relax (2s)': '#7f8c8d' # Gray
    }
    colors = [color_map.get(node, '#95a5a6') for node in emotion_counts.index]

    ax1.pie(
        emotion_counts, 
        labels=emotion_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    ax1.set_title("Time Distribution of Human Emotional States", fontsize=12, fontweight='bold', pad=10)

    # --- GRAPH 2: Chronological Line Chart of Behavioral Metrics ---
    # Convert string timestamps into workable chronological datetime markers
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    ax2.plot(df['Timestamp'], df['Mouth_Ratio'], marker='o', color='#8e44ad', linewidth=2, label="Mouth Open Ratio")
    ax2.plot(df['Timestamp'], df['Focus_Metric'], marker='s', color='#d35400', linewidth=1.5, linestyle='--', label="Brow Deviation (Anger/Stress)")
    
    ax2.set_title("Biometric Sensor Waveforms Over Time", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Time Coordinates Logged", fontsize=10)
    ax2.set_ylabel("Geometric Mathematical Delta Scale", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc="upper right")
    
    # Rotate timestamps diagonally so they don't overlap on the horizontal axis
    plt.xticks(rotation=30, ha='right')

    # Clean layout wrapping and render the panel window
    plt.tight_layout()
    print("[RENDER COMPLETE] Displaying dashboard engine view matrix...")
    plt.show()

if __name__ == "__main__":
    generate_biometric_dashboard()