

# Clustering Algorithms

Clustering algorithms are unsupervised learning techniques used to automatically group data points based on similarity. Unlike supervised learning, clustering doesn’t rely on predefined labels. Instead, it explores the structure of the data to discover meaningful groupings (clusters) based on feature patterns. Clustering is often used alongside dimensionality reduction techniques like **PCA** or **t-SNE** to visualize complex high-dimensional relationships in a lower-dimensional space.




## What they are used for

Clustering algorithms are powerful tools for:

- **Customer segmentation**: Grouping users or stores by behavior or demographics (e.g., segmenting retail stores by size, sales, and geography)
- **Anomaly detection**: Identifying outliers by looking at how isolated they are from any cluster
- **Market basket analysis**: Finding purchasing patterns across similar groups of customers or products

These applications are especially common when no labeled data is available, and the goal is to explore or categorize data in an interpretable way.




## Types of Algorithms

There are several families of clustering algorithms, each with different assumptions about the data structure:

- **Partitioning-based** (e.g., K-Means): Assigns data into a fixed number of clusters by minimizing variance.
- **Hierarchical**: Builds a tree of clusters by iteratively merging or splitting groups based on distance.
- **Density-based** (e.g., DBSCAN): Groups points that are closely packed together and labels sparse regions as outliers.
- **Model-based** (e.g., GMM): Assumes data is generated from a mixture of probabilistic distributions.

Each algorithm comes with trade-offs depending on your dataset shape, dimensionality, and noise.




# K-Means

A simple and popular partitioning method that:

- Groups data by minimizing the **within-cluster sum of squares (WCSS)**
- Assumes **spherical clusters** and equal variance
- Requires the number of clusters (**K**) to be defined in advance
- Works well on linearly separable and evenly sized clusters

Internally, it uses **Euclidean distance** in high-dimensional space to assign points to the closest centroid and iteratively updates cluster centers.

**Common enhancements**:

- `k-means++` initialization (for smarter seeding)
- Elbow method or Silhouette score to choose K



## Let's deep dive into the marvelous linear algebra behind K-Means.

To truly understand how K-Means works, it’s helpful to look at it from a linear algebra perspective. At its core, K-Means iteratively minimizes the **Within-Cluster Sum of Squares (WCSS)**, a form of squared Euclidean distance that can be expressed using vector operations.

The process can be interpreted as:

1. Projecting points into a **K-dimensional vector space**, where each cluster center is a vector.
2. Assigning each point to the cluster whose **centroid vector** has the minimum Euclidean distance to it.
3. Recalculating centroids as the **mean vector** of all points currently assigned to each cluster.

This repeats until convergence - i.e., assignments no longer change or the WCSS stops decreasing significantly.

This mathematical framework allows K-Means to be both simple and fast, especially in high-dimensional data like store segmentation, where human visualization isn't possible but algebraic representations shine.



### Context: Store Clustering

Imagine you have a dataset like this for each store, where each row is a high-dimensional vector representing the store’s characteristics. These vectors form a **cloud of points** in a 4D space. K-Means acts like a gravitational system, pulling these points toward central locations (centroids) until each group stabilizes.

| Store ID | Monthly Purchase | #Categories | Store Size (m²) | region_1 | region_2 |
| -------- | ---------------- | ----------- | --------------- | --------- | --------- |
| 001      | 8,000            | 4           | 100             | 1    | 0      |
| 002      | 3,000            | 2           | 50              | 0   | 1      |
| ...      | ...              | ...         | ...             | ...  | ...     |



Each of these values becomes a **dimension** in the clustering algorithm. Even though you can’t visualize this 4D space, the algorithm processes it just like 2D or 3D — just with more features.




### Task:

We want to group stores into clusters like:

- Cluster A: Big stores buying a lot of products across many categories.
- Cluster B: Small convenience stores buying little.
- Cluster C: Medium-size stores in suburban cities with average sales.

To achieve that, K-Means looks for **groupings with minimal internal variance**. In practice, this gives rise to insights that help with operational planning, targeted marketing, and supply chain optimization. **These clusters are interpretable**, data-driven segments — not arbitrary groupings.




### Actions:

#### 1. **Standardize the Data**

Why? Because features like purchase value and store size are on different scales.

**Standardization formula**:
```mathematica
z = (x - μ)/σ
```
This centers the data (mean = 0, std = 1), preventing large features (e.g., purchase $) from dominating.



#### 2. **Initialize K Centroids**

Let’s say you pick **K = 3**.

Think of this as placing 3 random "magnets" in the feature space.

##### **So how are these magnets placed under the hood?**

The default (naive) approach picks **K random data points** as initial centroids — but this can lead to poor or slow convergence if those points are too close or not representative.

**`k-means++`: Smarter Initialization**

`k-means++` improves this by spreading out initial centroids more intelligently:

1. Randomly select the first centroid from the data.
2. For each remaining point x, compute its distance squared from the nearest already chosen centroid.
3. Choose the next centroid with probability proportional to that distance (further points are more likely to be chosen).
4. Repeat until you have K centroids.

##### Numerical Example

Suppose your stores (after standardization) are:

- A: [0.5, 0.2]
- B: [-0.6, -0.4]
- C: [0.9, 1.0]
- D: [0.3, -0.2]
- E: [-0.8, 0.6]

Let’s initialize K = 2:

- Randomly select A as the first centroid.
- Compute distances to A:
  - B: d² ≈ 1.17
- C: d² ≈ 1.34
- D: d² ≈ 0.20
- E: d² ≈ 1.70

Now we sample the second centroid from B, C, D, or E - but **E is most likely** since it’s furthest from A.

This leads to better-separated starting points and helps avoid poor local minima.



#### 3. **Assign Points to Closest Centroid**

Each store is assigned to the cluster with the nearest centroid using **Euclidean distance**:

**Euclidean Distance formula** between a store xᵢ and centroid μₖ:
```mathematica
d[xᵢ, μₖ] = Sqrt[(xᵢ^(1) - μₖ^(1))² + (xᵢ^(2) - μₖ^(2))² + ...]
```

##### Under the Hood — Practical Example

Let’s say:

- Store A (standardized): 
  ```mathematica
  x = {0.5, 0.2}
  ```
  
- Centroid 1: 
  ```mathematica
  μ₁ = {0.3, 0.1}
  ```
  
- Centroid 2:
  ```mathematica
  μ₂ = {-0.6, -0.4}
  ```
  

Compute distances:
```mathematica
d[x, μ₁] = Sqrt[(0.5 - 0.3)² + (0.2 - 0.1)²] = Sqrt[0.04 + 0.01] = Sqrt[0.05] ≈ 0.22
```

```mathematica
d[x, μ₂] = Sqrt[(0.5 + 0.6)² + (0.2 + 0.4)²] = Sqrt[1.21 + 0.36] = Sqrt[1.57] ≈ 1.25
```

**Store A is assigned to Centroid 1**.

This process repeats for every store in the dataset at every iteration of the algorithm.

#### A little bit of math: What is Euclidean Distance?

Euclidean distance is just a generalization of the Pythagorean theorem into higher dimensions.

**Formula in n-dimensional space:**
```mathematica
d[x, y] = Sqrt[Sum[(xᵢ - yᵢ)², {i, 1, n}]]
```

Where:

```mathematica
x = {x₁, x₂, ..., xₙ}
```

```mathematica
y = {y₁, y₂, ..., yₙ}
```



**n** is the number of features (dimensions)

Even in a **4D, 7D, or 100D space**, the math remains the same. You're just summing more squared differences.



#### 4. **Recalculate the Centroids**

Once every store has been assigned to the closest centroid, we need to **recalculate the position of each centroid** to reflect the new center of the stores that now belong to its cluster.

Why? Because each centroid should represent the "average" position of the stores currently assigned to it. This step ensures that centroids gradually **drift toward the true center of mass** of their cluster — leading to better groupings in the next round.

**Centroid formula**:
```mathematica
μₖ = (1/Nₖ) * Sum[xᵢ, {i, 1, Nₖ}]
```
Where:

- Nₖ is the number of stores in cluster k
- xᵢ are the data points in cluster k

##### Numerical Example

Let’s assume you’re clustering stores in 2D space based on standardized features: purchase value and store size.

After assigning points in Step 3, suppose:

- Cluster 1 has Stores A and C:
  - A: [0.5, 0.2]
  - C: [0.3, -0.1]

Now recalculate the centroids:

- **Centroid 1** (for Cluster 1):
  ```mathematica
  μ₁ = (1/2) * ({0.5, 0.2} + {0.3, -0.1}) = (1/2) * {0.8, 0.1} = {0.4, 0.05}
  ```

This shift in centroids better represents the current groupings. In the next iteration, these updated centroids will be used to reassign the stores again.



#### 5. **Repeat Steps 3 & 4 Until Convergence**

The K-Means algorithm **loops** through Steps 3 and 4 — assignment and centroid update — until one of two stopping conditions is met:

- **Convergence**: Cluster assignments no longer change.
- **Centroid shift is small**: Movement of centroids is below a chosen threshold (e.g., tolerance = 0.0001).

------

##### Understanding the Cost Function

The algorithm is minimizing this cost function:
```mathematica
J = Sum[Sum[Norm[xᵢ - μₖ]², {xᵢ ∈ Cₖ}], {k, 1, K}]
```
Where:

- xᵢ: A store (data point)
- μₖ: The centroid of cluster k
- ‖xᵢ - μₖ‖²: The squared Euclidean distance between a store and its cluster center
- Cₖ: The set of stores assigned to cluster k

This function measures **how tight the clusters are** — the total sum of squared distances from each point to its centroid.

##### Numerical Example

Suppose Cluster 1 has:

- A: [0.5, 0.2]
- C: [0.3, -0.1]
- Centroid: [0.4, 0.05]

Calculate WCSS (within-cluster sum of squares) for Cluster 1:

- Distance A to centroid:
  ```mathematica
  Norm[{0.5, 0.2} - {0.4, 0.05}]² = (0.1)² + (0.15)² = 0.01 + 0.0225 = 0.0325
  ```

- Distance C to centroid:
  ```mathematica
  Norm[{0.3, -0.1} - {0.4, 0.05}]² = (-0.1)² + (-0.15)² = 0.01 + 0.0225 = 0.0325
  ```

Total cost for Cluster 1:
```mathematica
J₁ = 0.0325 + 0.0325 = 0.065
```
The algorithm will attempt to **minimize the total J across all clusters**, iterating until improvement stops.

------



#### Summary of the Math (All in One View)

| Concept             | Math                                                        | How It’s Used in K-Means                                     |
| ------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| **Distance**        | d[xᵢ, μₖ] = Sqrt[Sum[(xᵢ^(j) - μₖ^(j))², {j}]] | Measures how close a store is to a centroid for cluster assignment |
| **Centroid Update** | μₖ = (1/Nₖ) * Sum[xᵢ, {xᵢ ∈ Cₖ}]                    | Repositions the centroid to reflect the average of its assigned stores |
| **Cost Function**   | J = Sum[Sum[Norm[xᵢ - μₖ]², {xᵢ ∈ Cₖ}], {k}]           | The objective function minimized through iterative reassignments |



## Limitations to Keep in Mind

- K-Means assumes **spherical clusters** - not great if your stores form irregular shapes in the feature space.
- You must **predefine K** (although you can use methods like the Elbow Method to find it).
- It’s sensitive to **initialization** (which is why `k-means++` was introduced).


