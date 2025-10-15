import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_nth_latest_subfolder(parent_folder, n=1):
    subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    if len(subfolders) < n:
        return None
    # Sort by modification time (most recent last)
    subfolders.sort(key=os.path.getmtime)
    return subfolders[-n]

# Example usage:
latest_outputs = get_nth_latest_subfolder("outputs", n=2)
latest_outputs_test = get_nth_latest_subfolder("outputs_test", n=1)
print("Third most recent subfolder:", latest_outputs)
print("Third most recent subfolder:", latest_outputs_test)

# Find latest folders
outputs_folder = "outputs"
outputs_test_folder = "outputs_test"


# Image filenames
img_names = [
    "rewards_metrics.png",
    "rewards_smoothed.png",
    "training_loss.png",
    "training_positions.png"
]
img_paths = [os.path.join(latest_outputs, name) for name in img_names]

# Score progression from outputs_test
score_img = os.path.join(latest_outputs_test, "score_progression.png")
img_paths.append(score_img)
reward = "Defense"
abbr = "def"
# External images (update these paths as needed)
# DQL METRIC IMAGES
external_img1 = f"/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/MetricsPlots/DQL_{abbr}/__{reward}_comparison.png"
external_img2 = f"/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/MetricsPlots/DQL_{abbr}/boxplot_{reward}.png"
external_img3 = f"/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/MetricsPlots/DQL_{abbr}/all_metric_comparisons.png"

# DQL IMAGES
# external_img1 = f"/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/MetricsPlots/DQL_/__{reward}_comparison.png"
# external_img2 = f"/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/MetricsPlots/DQL_/boxplot_{reward}.png"
# external_img3 = f"/usr/local/src/robot/cognitiveInteraction/MetricsChefsHat/MetricsPlots/DQL_/all_metric_comparisons.png"

img_paths.extend([external_img1, external_img2, external_img3])

# Load images
images = []
for path in img_paths:
    if os.path.exists(path):
        images.append(mpimg.imread(path))
    else:
        print(f"Image not found: {path}")
        images.append(None)

# Update titles for all images
titles = img_names + ["score_progression", "multiple_plots_single_metric", f"boxplot_{reward}", "all_metric_comparisons"]

# Plot all images in a single figure
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle(f'DQL {abbr} Metrics Comparison (First row Training, Second row Testing)', fontsize=20)
axes = axes.flatten()

for ax, img, title in zip(axes, images, titles):
    if img is not None:
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    else:
        ax.set_title(f"{title}\n[Not found]")
        ax.axis('off')

plt.tight_layout()
plt.savefig(f"combined_results_{reward}.png")
plt.close()
print(f"Combined image saved as combined_results_{reward}.png")