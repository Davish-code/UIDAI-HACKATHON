import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

# Set professional style
sns.set_theme(style="whitegrid")

# ==========================================
# 1. LOAD & PREP DATA (ETL Pipeline)
# ==========================================
print("🚀 Starting AI Analysis Engine...")
files = [
    'api_data_aadhar_enrolment_0_500000.csv',
    'api_data_aadhar_enrolment_500000_1000000.csv',
    'api_data_aadhar_enrolment_1000000_1006029.csv'
]

dfs = []
for f in files:
    try:
        dfs.append(pd.read_csv(f))
    except:
        pass
df = pd.concat(dfs, ignore_index=True)
df.drop_duplicates(inplace=True)

# Feature Engineering
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
df['Month'] = df['date'].dt.to_period('M').astype(str)
df['Total_Activity'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']

# ==========================================
# 2. AI TOOL #1: K-MEANS CLUSTERING
# ==========================================
# Goal: Use ML to automatically label districts based on their "Personality"
print("🤖 Running K-Means Clustering on Districts...")

# Prepare District Data
district_profile = df.groupby(['state', 'district'])[['age_0_5', 'age_5_17', 'Total_Activity']].sum().reset_index()
district_profile['Student_Share'] = district_profile['age_5_17'] / district_profile['Total_Activity']
district_profile['Birth_Share'] = district_profile['age_0_5'] / district_profile['Total_Activity']

# Select Features for ML (Birth Ratio vs Student Ratio)
X_cluster = district_profile[['Birth_Share', 'Student_Share']]
X_cluster = X_cluster.fillna(0) # Handle potential 0/0 division

# Initialize AI Model (3 Clusters: School Hubs, Birthing Hubs, Balanced)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
district_profile['Cluster_Label'] = kmeans.fit_predict(X_cluster)

# Map generic labels (0,1,2) to meaningful names based on centroids
centroids = kmeans.cluster_centers_
# Logic: The cluster with highest 'Student_Share' centroid is 'School Hub'
student_col_idx = 1 # Student_Share is 2nd column
cluster_map = {
    np.argmax(centroids[:, student_col_idx]): 'School Hub (Target)',
    np.argmax(centroids[:, 0]): 'Birthing Center',
}
# Fill remaining as 'Balanced'
unique_labels = set([0,1,2])
used_labels = set(cluster_map.keys())
remaining = list(unique_labels - used_labels)[0]
cluster_map[remaining] = 'Balanced/Mixed'

district_profile['Cluster_Name'] = district_profile['Cluster_Label'].map(cluster_map)

# ==========================================
# 3. AI TOOL #2: ISOLATION FOREST
# ==========================================
# Goal: Use 'Isolation Forest' (Fraud Detection ML) to find anomalies
print("🕵️ Running Isolation Forest for Anomaly Detection...")

pincode_stats = df.groupby('pincode')['Total_Activity'].sum().reset_index()
X_iso = pincode_stats[['Total_Activity']]

# Initialize Model (contamination=0.001 means we look for top 0.1% rarest events)
iso_forest = IsolationForest(contamination=0.001, random_state=42)
pincode_stats['Anomaly_Flag'] = iso_forest.fit_predict(X_iso)

# Filter for Anomalies (-1 indicates outlier)
ml_anomalies = pincode_stats[pincode_stats['Anomaly_Flag'] == -1].sort_values(by='Total_Activity', ascending=False).head(5)
ml_anomalies['pincode'] = ml_anomalies['pincode'].astype(str)

# ==========================================
# 4. VISUALIZATION DASHBOARD
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
plt.subplots_adjust(hspace=0.4)

# Graph 1: Trend
monthly_trend = df.groupby('Month')['Total_Activity'].sum().reset_index()
sns.lineplot(data=monthly_trend, x='Month', y='Total_Activity', marker='o', ax=axes[0,0], color='#2c3e50')
axes[0,0].set_title('Temporal Trend (Seasonality)', fontsize=14, fontweight='bold')

# Graph 2: ML CLUSTERS (The AI Visual)
sns.scatterplot(
    data=district_profile, x='age_0_5', y='age_5_17', hue='Cluster_Name', 
    size='Total_Activity', sizes=(20, 400), alpha=0.7, ax=axes[0,1], palette='deep'
)
axes[0,1].set_title('AI Segmentation: K-Means Clustering', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('New Births (0-5)')
axes[0,1].set_ylabel('School Updates (5-17)')

# Graph 3: Priority List (From ML Cluster)
# Get top districts specifically from the "School Hub" cluster
school_targets = district_profile[district_profile['Cluster_Name'] == 'School Hub (Target)'].sort_values(by='Total_Activity', ascending=False).head(10)
sns.barplot(data=school_targets, x='Student_Share', y='district', palette='Reds', ax=axes[1,0])
axes[1,0].set_title('Top "School Hub" Districts (Identified by ML)', fontsize=14, fontweight='bold')

# Graph 4: ML Anomalies
sns.barplot(data=ml_anomalies, x='Total_Activity', y='pincode', color='salmon', ax=axes[1,1])
axes[1,1].set_title('AI-Detected Anomalies (Isolation Forest)', fontsize=14, fontweight='bold')

print("✅ Analysis Complete.")
print(f"Districts classified as School Hubs: {len(district_profile[district_profile['Cluster_Name'] == 'School Hub (Target)'])}")
print(f"Anomalies Detected by AI: {len(ml_anomalies)}")

plt.show()
