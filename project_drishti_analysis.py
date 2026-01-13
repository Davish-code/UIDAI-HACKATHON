import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import time
import sys
import webbrowser  # <--- NEW: Allows us to control the browser
import os          # <--- NEW: Helps find the file path

# Advanced AI Libraries
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# API & Mapping Libraries
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import folium

# Set professional style
sns.set_theme(style="whitegrid")

# ==========================================
# 0. UI UTILITIES (Hacker Style)
# ==========================================
class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def typing_print(text, speed=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    print()

print(Color.BOLD + "🚀 INITIATING PROJECT DRISHTI AI ENGINE..." + Color.END)

# ==========================================
# 1. DATA LOADING
# ==========================================
files = [
    'api_data_aadhar_enrolment_0_500000.csv',
    'api_data_aadhar_enrolment_500000_1000000.csv',
    'api_data_aadhar_enrolment_1000000_1006029.csv'
]

dfs = []
for f in files:
    try:
        try:
            dfs.append(pd.read_csv(f))
        except FileNotFoundError:
            dfs.append(pd.read_csv(f"data/{f}"))
        print(f"  {Color.GREEN}[OK]{Color.END} Loaded: {f}")
    except:
        pass

if not dfs:
    print(Color.RED + "❌ FATAL ERROR: No data files found." + Color.END)
    sys.exit()

df = pd.concat(dfs, ignore_index=True)
df.drop_duplicates(inplace=True)

# Feature Engineering
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
df['Month'] = df['date'].dt.to_period('M').astype(str)
df['Total_Activity'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']

# ==========================================
# 2. RUNNING AI MODELS
# ==========================================
print("\n" + Color.CYAN + "⚙️  RUNNING ANALYTICS PROTOCOLS..." + Color.END)

# --- A. Trend ---
monthly_trend = df.groupby('Month')['Total_Activity'].sum().reset_index()

# --- B. Clustering (K-Means) ---
district_profile = df.groupby(['state', 'district'])[['age_0_5', 'age_5_17', 'Total_Activity']].sum().reset_index()
district_profile['Student_Share'] = district_profile['age_5_17'] / district_profile['Total_Activity']
district_profile['Birth_Share'] = district_profile['age_0_5'] / district_profile['Total_Activity']

X_cluster = district_profile[['Birth_Share', 'Student_Share']].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
district_profile['Cluster_Label'] = kmeans.fit_predict(X_cluster)

centroids = kmeans.cluster_centers_
school_idx = np.argmax(centroids[:, 1]) 
def get_label(l): return 'School Hub (Target)' if l == school_idx else 'General/Birth'
district_profile['Cluster_Name'] = district_profile['Cluster_Label'].apply(get_label)

# --- C. Anomaly Detection ---
pincode_stats = df.groupby('pincode')['Total_Activity'].sum().reset_index()
iso = IsolationForest(contamination=0.001, random_state=42)
pincode_stats['Anomaly'] = iso.fit_predict(pincode_stats[['Total_Activity']])
anomalies = pincode_stats[pincode_stats['Anomaly'] == -1].sort_values(by='Total_Activity', ascending=False).head(5)
anomalies['pincode'] = anomalies['pincode'].astype(str)

# --- D. Optimization ---
targets = district_profile[district_profile['Cluster_Name'] == 'School Hub (Target)'].sort_values(by='Total_Activity', ascending=False).head(10).copy()
def calc_vans(load): return math.ceil((load / 30) / 50)
targets['Vans_Required'] = targets['Total_Activity'].apply(calc_vans)

# --- E. Benford's Law ---
def get_digit(n): return int(str(n)[0])
digit_data = df[df['Total_Activity'] > 10]['Total_Activity'].apply(get_digit)
digit_counts = digit_data.value_counts().sort_index()
benford_pct = [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6]
expected_counts = [len(digit_data) * (p/100) for p in benford_pct]

# ==========================================
# 3. DASHBOARD GENERATION (SHOW FIRST)
# ==========================================
print("  [i] Generating Command Dashboard...")
fig, axes = plt.subplots(3, 2, figsize=(20, 24))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 1. Trend
sns.lineplot(data=monthly_trend, x='Month', y='Total_Activity', marker='o', ax=axes[0,0], color='#2c3e50', linewidth=3)
axes[0,0].set_title('1. Temporal Trend: The "September Surge"', fontsize=14, fontweight='bold')

# 2. Clusters
sns.scatterplot(data=district_profile, x='age_0_5', y='age_5_17', hue='Cluster_Name', size='Total_Activity', sizes=(20, 300), ax=axes[0,1], palette='viridis')
axes[0,1].set_title('2. AI Segmentation: Births vs Students', fontsize=14, fontweight='bold')

# 3. Targets
sns.barplot(data=targets, x='Student_Share', y='district', palette='Reds', ax=axes[1,0])
axes[1,0].set_title('3. Top Priority School Districts', fontsize=14, fontweight='bold')

# 4. Anomalies
sns.barplot(data=anomalies, x='Total_Activity', y='pincode', color='salmon', ax=axes[1,1])
axes[1,1].set_title('4. AI-Detected Fraud Hotspots', fontsize=14, fontweight='bold')

# 5. Optimization
sns.barplot(data=targets, x='Vans_Required', y='district', palette='Blues_r', ax=axes[2,0])
axes[2,0].set_title('5. Resource Plan: Mobile Vans Required', fontsize=14, fontweight='bold')

# 6. Forensics
x_digits = range(1, 10)
axes[2,1].bar(x_digits, digit_counts, alpha=0.6, label='Actual Data', color='blue')
axes[2,1].plot(x_digits, expected_counts, color='red', marker='o', linewidth=2, label="Benford's Law (Expected)")
axes[2,1].set_title('6. Forensic Audit: Data Integrity', fontsize=14, fontweight='bold')
axes[2,1].legend()

# Save
plt.savefig('dashboard_combined.png')
print(f"  {Color.GREEN}✅ Dashboard Saved: dashboard_combined.png{Color.END}")
print(f"  {Color.YELLOW}[ACTION REQUIRED] Close the graph window to continue...{Color.END}")

# Show Graphs Immediately
plt.show()

# ==========================================
# 4. INTERACTIVE "HACKER-STYLE" PROMPT
# ==========================================
print("\n" + Color.CYAN + "="*60 + Color.END)
print(Color.BOLD + "           🗺️  GEOSPATIAL INTELLIGENCE MODULE" + Color.END)
print(Color.CYAN + "="*60 + Color.END)

print(Color.YELLOW + "  [?] System requires authorization to access Satellite API." + Color.END)
print("      This will generate an interactive HTML deployment map.")

user_response = input(f"\n  {Color.BOLD}>> Initiate Mapping Protocol? (Y/N): {Color.END}").strip().lower()

if user_response in ['y', 'yes']:
    print("\n" + Color.GREEN + "  [✓] Authorization Granted." + Color.END)
    typing_print("  [i] Establishing connection with Nominatim Servers...", 0.03)
    
    geolocator = Nominatim(user_agent="project_drishti_hackathon")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")

    def get_lat_long(district_name):
        try:
            location = geolocator.geocode(f"{district_name}, India", timeout=10)
            return (location.latitude, location.longitude) if location else (None, None)
        except:
            return None, None

    print(f"  {Color.CYAN}[Wait] Processing {len(targets)} High-Priority Targets:{Color.END}")
    
    for index, row in targets.iterrows():
        dist = row['district']
        vans = row['Vans_Required']
        
        lat, lng = get_lat_long(dist)
        
        if lat and lng:
            popup_text = f"<b>District:</b> {dist}<br><b>Vans:</b> {vans}"
            folium.Marker([lat, lng], popup=popup_text, icon=folium.Icon(color='red')).add_to(m)
            print(f"    {Color.GREEN}✓{Color.END} Geocoded: {dist}")
        else:
             print(f"    {Color.RED}✗{Color.END} Failed: {dist}")

    # SAVE AND AUTO-OPEN
    m.save("deployment_map.html")
    print("\n" + Color.BOLD + Color.GREEN + "✅ SUCCESS: Map Generated." + Color.END)
    
    # -----------------------------------------------
    # NEW CODE: AUTOMATICALLY OPEN IN BROWSER
    # -----------------------------------------------
    print(f"  {Color.CYAN}[i] Launching Interface in Default Browser...{Color.END}")
    try:
        map_path = os.path.abspath("deployment_map.html")
        webbrowser.open('file://' + map_path)
    except:
        print(f"  {Color.YELLOW}[!] Could not auto-launch. Please open 'deployment_map.html' manually.{Color.END}")

else:
    print("\n" + Color.RED + "  [X] Protocol Aborted by User." + Color.END)

# ==========================================
# 5. FINAL REPORT
# ==========================================
print("\n" + Color.BOLD + "📊 EXECUTIVE SUMMARY" + Color.END)
print("-" * 40)
print(f"1. {Color.BOLD}Optimization:{Color.END} {len(targets)} Districts identified for immediate deployment.")
print(f"2. {Color.BOLD}Security:{Color.END} {len(anomalies)} Critical Fraud Anomalies detected.")
print(f"3. {Color.BOLD}Forensics:{Color.END} Benford's Law Analysis completed.")
print("-" * 40)
print(Color.CYAN + "Process Terminated." + Color.END)
