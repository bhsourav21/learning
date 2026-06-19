import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import PCA

load_dotenv()

client = OpenAI()

texts = [
    "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai",
    "Dominos Pizza is the Best",
    "Pizza Hut is another famous pizza chain",
    "My favourite subject is mathematics",
    "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour",
    "Sourav Ganguly is the best Indian cricket captain of all time",
    "I enjoy cooking non-veg Indian food",
    "Reading story books is an excellent habit",
    "Chetan Bhagat's novels attract me a lot",
    "Mango is the best fruit"
]

response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

vectors = np.array([item.embedding for item in response.data])

print(f"\nTotal embeddings received: {len(vectors)}\n")

coords = PCA(n_components=2).fit_transform(vectors)

plt.figure(figsize=(12, 8))
plt.scatter(coords[:, 0], coords[:, 1], color="steelblue", s=80, zorder=2)

for i, text in enumerate(texts):
    plt.annotate(
        f"{i+1}. {text}",
        xy=(coords[i, 0], coords[i, 1]),
        xytext=(8, 4),
        textcoords="offset points",
        fontsize=8,
        wrap=True
    )

plt.title("2D PCA Visualisation of Text Embeddings", fontsize=13)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("embedding_2D_visualize.png", dpi=150)
plt.show()
print("Plot saved to embedding_2D_visualize.png")

