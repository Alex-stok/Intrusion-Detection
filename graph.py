import matplotlib.pyplot as plt

# Example class counts (replace with your actual values if different)
labels = ['BENIGN: 499590', 'DrDoS_NTP: 497064', 'DrDoS_UDP: 493481', 'Syn: 436053']
sizes = [499590, 497064, 493481, 436053]

colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99']
explode = (0.05, 0, 0, 0)  # explode BENIGN for emphasis

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, explode=explode)
plt.title('Training Data Distribution')
plt.tight_layout()
plt.show()