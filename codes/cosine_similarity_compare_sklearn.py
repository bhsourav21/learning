import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

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
    "Chetan Bhagat's novels attact me a lot",
    "Mangoe is the best fruit"
]

response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

vectors = np.array([item.embedding for item in response.data])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\nTotal embeddings received: {len(vectors)}\n")
print("Cosine Similarity for each pair:\n")
print(f"{'#':<5} {'Input A':<8} {'Input B':<8} {'Similarity':<10} Texts")
print("-" * 100)

pair = 1
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        score = cosine_similarity(vectors[i], vectors[j])
        print(f"{pair:<5} {i+1:<8} {j+1:<8} {score:<10.4f} \"{texts[i]}\" vs \"{texts[j]}\"")
        pair += 1

# --- sklearn implementation ---
print("\n" + "=" * 100)
print("Cosine Similarity using sklearn:\n")
print(f"{'#':<5} {'Input A':<8} {'Input B':<8} {'Similarity':<10} Texts")
print("-" * 100)

# sklearn expects 2D arrays; cosine_similarity returns an (n x n) matrix
similarity_matrix = sklearn_cosine_similarity(vectors)

pair = 1
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        score = similarity_matrix[i][j]
        print(f"{pair:<5} {i+1:<8} {j+1:<8} {score:<10.4f} \"{texts[i]}\" vs \"{texts[j]}\"")
        pair += 1

# --- Verification: confirm both approaches produce matching results ---
print("\n" + "=" * 100)
print("Verification: NumPy vs sklearn results\n")

all_match = True
max_diff = 0.0
pair = 1
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        numpy_score = cosine_similarity(vectors[i], vectors[j])
        sklearn_score = similarity_matrix[i][j]
        diff = abs(numpy_score - sklearn_score)
        max_diff = max(max_diff, diff)
        if diff > 1e-6:
            print(f"MISMATCH at pair {pair} (texts {i+1} vs {j+1}): NumPy={numpy_score:.8f}, sklearn={sklearn_score:.8f}, diff={diff:.2e}")
            all_match = False
        pair += 1

if all_match:
    print(f"All {pair - 1} pairs match! Max absolute difference: {max_diff:.2e}")


# Output:
# ----------
# Total embeddings received: 10

# Cosine Similarity for each pair:

# #     Input A  Input B  Similarity Texts
# ----------------------------------------------------------------------------------------------------
# 1     1        2        0.0434     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "Dominos Pizza is the Best"
# 2     1        3        0.1055     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "Pizza Hut is another famous pizza chain"
# 3     1        4        0.1049     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "My favourite subject is mathematics"
# 4     1        5        0.1142     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour"
# 5     1        6        0.3287     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "Sourav Ganguly is the best Indian cricket captain of all time"
# 6     1        7        0.1730     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "I enjoy cooking non-veg Indian food"
# 7     1        8        0.0926     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "Reading story books is an excellent habit"
# 8     1        9        0.2762     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "Chetan Bhagat's novels attact me a lot"
# 9     1        10       0.0627     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "Mangoe is the best fruit"
# 10    2        3        0.5728     "Dominos Pizza is the Best" vs "Pizza Hut is another famous pizza chain"
# 11    2        4        0.1712     "Dominos Pizza is the Best" vs "My favourite subject is mathematics"
# 12    2        5        0.0861     "Dominos Pizza is the Best" vs "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour"
# 13    2        6        0.2516     "Dominos Pizza is the Best" vs "Sourav Ganguly is the best Indian cricket captain of all time"
# 14    2        7        0.1802     "Dominos Pizza is the Best" vs "I enjoy cooking non-veg Indian food"
# 15    2        8        0.1161     "Dominos Pizza is the Best" vs "Reading story books is an excellent habit"
# 16    2        9        0.1088     "Dominos Pizza is the Best" vs "Chetan Bhagat's novels attact me a lot"
# 17    2        10       0.3051     "Dominos Pizza is the Best" vs "Mangoe is the best fruit"
# 18    3        4        0.1244     "Pizza Hut is another famous pizza chain" vs "My favourite subject is mathematics"
# 19    3        5        0.0322     "Pizza Hut is another famous pizza chain" vs "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour"
# 20    3        6        0.1135     "Pizza Hut is another famous pizza chain" vs "Sourav Ganguly is the best Indian cricket captain of all time"
# 21    3        7        0.1873     "Pizza Hut is another famous pizza chain" vs "I enjoy cooking non-veg Indian food"
# 22    3        8        0.1282     "Pizza Hut is another famous pizza chain" vs "Reading story books is an excellent habit"
# 23    3        9        0.1681     "Pizza Hut is another famous pizza chain" vs "Chetan Bhagat's novels attact me a lot"
# 24    3        10       0.1491     "Pizza Hut is another famous pizza chain" vs "Mangoe is the best fruit"
# 25    4        5        0.0408     "My favourite subject is mathematics" vs "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour"
# 26    4        6        0.1193     "My favourite subject is mathematics" vs "Sourav Ganguly is the best Indian cricket captain of all time"
# 27    4        7        0.2302     "My favourite subject is mathematics" vs "I enjoy cooking non-veg Indian food"
# 28    4        8        0.2071     "My favourite subject is mathematics" vs "Reading story books is an excellent habit"
# 29    4        9        0.2701     "My favourite subject is mathematics" vs "Chetan Bhagat's novels attact me a lot"
# 30    4        10       0.1855     "My favourite subject is mathematics" vs "Mangoe is the best fruit"
# 31    5        6        0.0384     "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour" vs "Sourav Ganguly is the best Indian cricket captain of all time"
# 32    5        7        0.0558     "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour" vs "I enjoy cooking non-veg Indian food"
# 33    5        8        0.1191     "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour" vs "Reading story books is an excellent habit"
# 34    5        9        0.1003     "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour" vs "Chetan Bhagat's novels attact me a lot"
# 35    5        10       0.1064     "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour" vs "Mangoe is the best fruit"
# 36    6        7        0.2088     "Sourav Ganguly is the best Indian cricket captain of all time" vs "I enjoy cooking non-veg Indian food"
# 37    6        8        0.1014     "Sourav Ganguly is the best Indian cricket captain of all time" vs "Reading story books is an excellent habit"
# 38    6        9        0.1807     "Sourav Ganguly is the best Indian cricket captain of all time" vs "Chetan Bhagat's novels attact me a lot"
# 39    6        10       0.2643     "Sourav Ganguly is the best Indian cricket captain of all time" vs "Mangoe is the best fruit"
# 40    7        8        0.2004     "I enjoy cooking non-veg Indian food" vs "Reading story books is an excellent habit"
# 41    7        9        0.2657     "I enjoy cooking non-veg Indian food" vs "Chetan Bhagat's novels attact me a lot"
# 42    7        10       0.2508     "I enjoy cooking non-veg Indian food" vs "Mangoe is the best fruit"
# 43    8        9        0.3173     "Reading story books is an excellent habit" vs "Chetan Bhagat's novels attact me a lot"
# 44    8        10       0.2041     "Reading story books is an excellent habit" vs "Mangoe is the best fruit"
# 45    9        10       0.1491     "Chetan Bhagat's novels attact me a lot" vs "Mangoe is the best fruit"

# ====================================================================================================
# Cosine Similarity using sklearn:

# #     Input A  Input B  Similarity Texts
# ----------------------------------------------------------------------------------------------------
# 1     1        2        0.0434     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "Dominos Pizza is the Best"
# 2     1        3        0.1055     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "Pizza Hut is another famous pizza chain"
# 3     1        4        0.1049     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "My favourite subject is mathematics"
# 4     1        5        0.1142     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour"
# 5     1        6        0.3287     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "Sourav Ganguly is the best Indian cricket captain of all time"
# 6     1        7        0.1730     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "I enjoy cooking non-veg Indian food"
# 7     1        8        0.0926     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "Reading story books is an excellent habit"
# 8     1        9        0.2762     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "Chetan Bhagat's novels attact me a lot"
# 9     1        10       0.0627     "Sachin scored 103 not out vs England while Chasing more than 350 in 4th innings in Chennai" vs "Mangoe is the best fruit"
# 10    2        3        0.5728     "Dominos Pizza is the Best" vs "Pizza Hut is another famous pizza chain"
# 11    2        4        0.1712     "Dominos Pizza is the Best" vs "My favourite subject is mathematics"
# 12    2        5        0.0861     "Dominos Pizza is the Best" vs "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour"
# 13    2        6        0.2516     "Dominos Pizza is the Best" vs "Sourav Ganguly is the best Indian cricket captain of all time"
# 14    2        7        0.1802     "Dominos Pizza is the Best" vs "I enjoy cooking non-veg Indian food"
# 15    2        8        0.1161     "Dominos Pizza is the Best" vs "Reading story books is an excellent habit"
# 16    2        9        0.1088     "Dominos Pizza is the Best" vs "Chetan Bhagat's novels attact me a lot"
# 17    2        10       0.3051     "Dominos Pizza is the Best" vs "Mangoe is the best fruit"
# 18    3        4        0.1244     "Pizza Hut is another famous pizza chain" vs "My favourite subject is mathematics"
# 19    3        5        0.0322     "Pizza Hut is another famous pizza chain" vs "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour"
# 20    3        6        0.1135     "Pizza Hut is another famous pizza chain" vs "Sourav Ganguly is the best Indian cricket captain of all time"
# 21    3        7        0.1873     "Pizza Hut is another famous pizza chain" vs "I enjoy cooking non-veg Indian food"
# 22    3        8        0.1282     "Pizza Hut is another famous pizza chain" vs "Reading story books is an excellent habit"
# 23    3        9        0.1681     "Pizza Hut is another famous pizza chain" vs "Chetan Bhagat's novels attact me a lot"
# 24    3        10       0.1491     "Pizza Hut is another famous pizza chain" vs "Mangoe is the best fruit"
# 25    4        5        0.0408     "My favourite subject is mathematics" vs "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour"
# 26    4        6        0.1193     "My favourite subject is mathematics" vs "Sourav Ganguly is the best Indian cricket captain of all time"
# 27    4        7        0.2302     "My favourite subject is mathematics" vs "I enjoy cooking non-veg Indian food"
# 28    4        8        0.2071     "My favourite subject is mathematics" vs "Reading story books is an excellent habit"
# 29    4        9        0.2701     "My favourite subject is mathematics" vs "Chetan Bhagat's novels attact me a lot"
# 30    4        10       0.1855     "My favourite subject is mathematics" vs "Mangoe is the best fruit"
# 31    5        6        0.0384     "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour" vs "Sourav Ganguly is the best Indian cricket captain of all time"
# 32    5        7        0.0558     "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour" vs "I enjoy cooking non-veg Indian food"
# 33    5        8        0.1191     "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour" vs "Reading story books is an excellent habit"
# 34    5        9        0.1003     "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour" vs "Chetan Bhagat's novels attact me a lot"
# 35    5        10       0.1064     "To get the maximum benefit of hybrid cars, you need to drive at around 60 miles/hour" vs "Mangoe is the best fruit"
# 36    6        7        0.2088     "Sourav Ganguly is the best Indian cricket captain of all time" vs "I enjoy cooking non-veg Indian food"
# 37    6        8        0.1014     "Sourav Ganguly is the best Indian cricket captain of all time" vs "Reading story books is an excellent habit"
# 38    6        9        0.1807     "Sourav Ganguly is the best Indian cricket captain of all time" vs "Chetan Bhagat's novels attact me a lot"
# 39    6        10       0.2643     "Sourav Ganguly is the best Indian cricket captain of all time" vs "Mangoe is the best fruit"
# 40    7        8        0.2004     "I enjoy cooking non-veg Indian food" vs "Reading story books is an excellent habit"
# 41    7        9        0.2657     "I enjoy cooking non-veg Indian food" vs "Chetan Bhagat's novels attact me a lot"
# 42    7        10       0.2508     "I enjoy cooking non-veg Indian food" vs "Mangoe is the best fruit"
# 43    8        9        0.3173     "Reading story books is an excellent habit" vs "Chetan Bhagat's novels attact me a lot"
# 44    8        10       0.2041     "Reading story books is an excellent habit" vs "Mangoe is the best fruit"
# 45    9        10       0.1491     "Chetan Bhagat's novels attact me a lot" vs "Mangoe is the best fruit"

# ====================================================================================================
# Verification: NumPy vs sklearn results

# All 45 pairs match! Max absolute difference: 5.55e-16