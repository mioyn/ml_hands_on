# Embeddings
### Semantic similarity
Semantic similarity is about measuring how close the meanings of words or sentences are, not just whether they look the same(capture meaningful relationships between words).
Before every words were represented seperatly(like giving each word a unique number i.e, one-hot encoding). Words like "king", "queen" and "car" are treated as totally separate. Semantic similarity is by recognizing that “king” and “queen” are similar in meaning because they are both royal titles.
We achieve semantic similarity by a technique called Embeddings

### Embeddings
Embeddings position words in a continuous vector space such that words with similar semantic or syntactic roles are located near each other (embeddings translate semantic similarity into geometric proximity in vector space)
    
  - Two words $$w_i$$​ and $$w_j$$​ (e.g., "king" and "queen")
  - $$v_i∈{R}^d$$ and $$v_j∈{R}^d$$ are embedding vector of d-dimensional 
  - Embedding model is trained so that $$sim(w_i​, w_j​)$$ (semantic similarity score) and $$prox(v_i, v_j)$$ (geometric proximity) align

```
OpenAI text embedding models:
text-embedding-3-large  →  3072 dimensions  →  Higher quality, better semantic resolution    
text-embedding-3-small  →  1536 dimensions  →  Cheaper, faster, very similar to older defaults
```
The "geometric proximity" of word or sentence embeddings is quantified using specific distance(Euclidean Distance) or similarity metrics(Cosine Similarity) in the vector space 

### Cosine Similarity
It measures the cosine of the angle between two vectors, ignoring their magnitude (length).
Single numeric value in the range [-1, 1]. 1 → identical direction (very similar), 0 → orthogonal (no similarity), and -1 → opposite direction (very dissimilar)

  ```math
  \text{sim}_{\text{cos}}(v_i, v_j) = \frac{v_i \cdot v_j}{\|v_i\|\|v_j\|} = \cos(\theta)
  ```
  Where θ is the angle between vectors **vᵢ** and **vⱼ**. Similar words have θ ≈ 0°, thus cos(θ) ≈ 1.

### Euclidean Distance
The most intuitive "straight-line" distance between two points in space. Similar words minimize this distance.
  ```math
  d_{\text{euclid}}(v_i, v_j) = \|v_i - v_j\|_2 = \sqrt{\sum_{k=1}^d (v_{i,k} - v_{j,k})^2}
  ```
  
