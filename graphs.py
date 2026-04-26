import matplotlib.pyplot as plt

# Data
bar_x = ["1st", "2nd", "3rd", "4th", "5th"]
bar_y = [34,65,23,76,43]

pie_data = [24, 65, 12, 34, 56]
pie_labels = ["amir", "bob", "charlie", "dave", "eve"]

scatter_x = ["amir", "bob", "charlie", "dave", "eve"]
scatter_y = [24, 65, 12, 34, 56]

hist_data = [10,20,20,30,30,30,40,50,60]

fig, ax = plt.subplots(1,2 , figsize=(13, 5))

# Plotting with corrected parameters
ax[0].bar(bar_x, bar_y, color='blue')
ax[0].set_title("Best matches score")
ax[0].set_xlabel("Match Rank")
ax[0].set_ylabel("Match Score")
ax[0].grid(axis='y',color='gray', linestyle='--', alpha=0.5)
# ax[0].text(
#     0.00, 0.95,
#     "User A has 100 1st matches, 50 2nd matches, and 25 3rd matches.",
#     transform=plt.gca().transAxes
# )

ax[1].pie(pie_data, autopct='%1.1f%%')
ax[1].set_title("Match Score Distribution")
ax[1].legend(pie_labels, loc="upper right")

plt.show()
