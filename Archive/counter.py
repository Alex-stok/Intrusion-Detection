import pandas as pd

file_path = "C:\\Users\\jahma\\OneDrive\\Desktop\\Datasets\\MachineLearningCSV\\Monday-WorkingHours.pcap_ISCX.csv"
df = pd.read_csv(file_path, low_memory=False)
df.columns = df.columns.str.strip()

benign_count = (df['Label'] == 'BENIGN').sum()
print(f"Number of BENIGN entries: {benign_count}")