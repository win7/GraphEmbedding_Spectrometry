from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_samples, silhouette_score

import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
import plotly.graph_objects as go
import plotly

colors = ["#FF00FF", "#3FFF00", "#00FFFF", "#FFF700", "#FF0000", "#0000FF", "#006600",
          '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', 'black',"gray"]
# colors = px.colors.sequential.Rainbow

def hello():
  print("Hello...")
  
def get_daltons(formula):
  d = {"C": 12, 
       "H": 1.00782503207, 
       "N": 14.0030740048,
       "O": 15.99491461956,
       "P": 30.97376163,
       "S": 31.972071,
       "Si": 27.9769265325,
       "F": 18.99840322,
       "Cl": 34.96885268,
       "Br": 78.9183371}

  f = ""
  n = ""
  daltons = 0
  try:
    for k in range(len(formula)):
      if formula[k].isnumeric():
        n += formula[k]
      else:
        if formula[k].isupper() and f != "":
          # print(f, n)
          if n == "":
            n = "1"
          daltons += d[f] * int(n)
          f = formula[k]
          n = ""
        else:
          f += formula[k]
    if n == "":
      n = "1"
    # print(f, n)
    daltons += d[f] * int(n)
  except:
    daltons = 0
  return daltons

def get_node_class(node_embeddings_2d1, node_embeddings_2d2):
  nodes = []
  classes1 = []
  classes2 = []

  for k, node_id in enumerate(node_embeddings_2d2.index):
    nodes.append(node_id)
    classes1.append(node_embeddings_2d2["labels"][k])
    if node_id in node_embeddings_2d1.index:
      index = list(node_embeddings_2d1.index).index(node_id)
      classes2.append(node_embeddings_2d1["labels"][index])
    else:
      classes2.append("X")
  data = {"Node id": nodes, "Class G1": classes1, "Class G2": classes2}
  df_match1 = pd.DataFrame(data=data)
  return df_match1.sort_values(by="Class G1", ascending=True)

def matching(node_embeddings_2d1, node_embeddings_2d2, node_embeddings_2d):
  n_classes = np.unique(node_embeddings_2d1["labels"].values)
  if -1 in n_classes:
    len1 = len(n_classes) - 1
  else:
    len1 = len(n_classes)
  
  labels1_ = node_embeddings_2d1["labels"].values.copy()
  size1_ = [8] * len(node_embeddings_2d1.index)
  opacity1_ = [0.3] * len(node_embeddings_2d1.index)
  for k, node_id in enumerate(node_embeddings_2d2.index):
    if node_id in node_embeddings_2d1.index and node_embeddings_2d2["labels"][k] != -1:
      index = list(node_embeddings_2d1.index).index(node_id)
      labels1_[index] = node_embeddings_2d2["labels"][k] + len1
      # size1_[index] = 10
      opacity1_[index] = 0.9
  node_embeddings_2d1["labels_"] = labels1_
  
  # Plot
  fig = make_subplots(rows=2, cols=2,
                      subplot_titles=("Group 1", "Group 2", "Group 1 - Group 2", "Group 1 + Group 2"),
                      horizontal_spacing=0.05, vertical_spacing=0.05)

  fig.add_trace(
      go.Scatter(
        x=node_embeddings_2d1.iloc[:, 0].values,
        y=node_embeddings_2d1.iloc[:, 1].values,
        mode="markers",
        name="markers",
        text=node_embeddings_2d1.index,
        hovertemplate="Node id: " + node_embeddings_2d1.index + "<br>Class: " + node_embeddings_2d1["labels"].astype(str),
        textposition="bottom center",
        showlegend=True,
        marker=dict(
          size=8,
          color=node_embeddings_2d1["labels"].values,
          opacity=0.9,
          colorscale=list(np.array(colors)[np.unique(node_embeddings_2d1["labels"])]), # "Rainbow",
          line_width=1
        ),
      ),
      row=1, col=1
  )

  colorscale_ = np.unique(node_embeddings_2d2["labels"])
  for k in range(len(colorscale_)):
    if colorscale_[k] != -1:
      colorscale_[k] +=  len1

  fig.add_trace(
      go.Scatter(
          x=node_embeddings_2d2.iloc[:, 0].values,
          y=node_embeddings_2d2.iloc[:, 1].values,
          mode="markers",
          name="markers",
          text=node_embeddings_2d2.index,
          hovertemplate="Node id: " + node_embeddings_2d2.index + "<br>Class: " + node_embeddings_2d2["labels"].astype(str),
          textposition="bottom center",
          showlegend=True,
          marker=dict(
            size=8,
            color=node_embeddings_2d2["labels"].values,
            opacity=0.9,
            colorscale=list(np.array(colors)[colorscale_]), # "Rainbow",
            line_width=1
          ),
          textfont=dict(
              family="sans serif",
              size=10,
              color="black"
          ),
      ),
      row=1, col=2
  )

  fig.add_trace(
      go.Scatter(
          x=node_embeddings_2d1.iloc[:, 0].values,
          y=node_embeddings_2d1.iloc[:, 1].values,
          mode="markers",
          name="markers",
          text=node_embeddings_2d1.index,
          hovertemplate="Node id: " + node_embeddings_2d1.index + "<br>Class: " + node_embeddings_2d1["labels_"].astype(str),
          textposition="bottom center",
          showlegend=True,
          marker=dict(
            size=8,
            color=labels1_,
            opacity=opacity1_,
            colorscale=list(np.array(colors)[np.unique(labels1_)]), # "Rainbow",
            line_width=1
          ),
      ),
      row=2, col=1
  ),

  fig.add_trace(
      go.Scatter(
          x=node_embeddings_2d.iloc[:, 0].values,
          y=node_embeddings_2d.iloc[:, 1].values,
          mode="markers",
          name="markers",
          text="Node id: " + node_embeddings_2d.index + "<br>Class: " + node_embeddings_2d["labels"].astype(str),
          hovertemplate="Node id: " + node_embeddings_2d.index + "<br>Class: " + node_embeddings_2d["labels"].astype(str),
          textposition="bottom center",
          showlegend=True,
          marker=dict(
            size=8,
            color=node_embeddings_2d["labels"].values,
            opacity=0.9,
            colorscale=list(np.array(colors)[np.unique(node_embeddings_2d["labels"])]), # "Rainbow",
            line_width=1
          ),
      ),
      row=2, col=2
  )

  fig.update_layout(height=1000, width=1000, title_text="Clustering Embeddings") # ,legend_tracegroupgap=50, showlegend=False)
  fig.show()

def visualization_cluster_embeddings(list_embeddings_2d):
  cols = 2
  rows = math.ceil(len(list_embeddings_2d) / cols)

  titles = []
  for k in range(len(list_embeddings_2d)):
    titles.append("Group {}".format(k + 1))
  
  fig = plotly.subplots.make_subplots(rows=rows, cols=cols,
                      subplot_titles=titles,
                      horizontal_spacing=0.05, vertical_spacing=0.05)

  for i, node_embeddings_2d in enumerate(list_embeddings_2d):
    fig.add_trace(
        go.Scatter(
            x=node_embeddings_2d.iloc[:, 0].values,
            y=node_embeddings_2d.iloc[:, 1].values,
            mode="markers",
            name="markers",
            text="Node id: " + node_embeddings_2d.index + "<br>Class: " + node_embeddings_2d["labels"].astype(str),
            textposition="bottom center",
            marker=dict(
              size=8,
              color=node_embeddings_2d["labels"].values,
              opacity=0.9,
              colorscale=list(np.array(colors)[np.unique(node_embeddings_2d["labels"])]), # "Rainbow",
              line_width=1
            ),
        ),
        row=math.ceil((i + 1) / cols), col=(i % cols) + 1
    )
  fig.update_layout(height=700*rows, width=1400, title_text="Embeddings", showlegend=True)
  fig.show()

def visualization_embeddings(list_embeddings_2d):
  cols = 2
  rows = math.ceil(len(list_embeddings_2d) / cols)

  titles = []
  for k in range(len(list_embeddings_2d)):
    titles.append("Group {}".format(k + 1))
  
  fig = plotly.subplots.make_subplots(rows=rows, cols=cols,
                      subplot_titles=titles,
                      horizontal_spacing=0.05, vertical_spacing=0.05)

  for i, node_embedding_2d in enumerate(list_embeddings_2d):
    fig.add_trace(
        go.Scatter(
            x=node_embedding_2d.iloc[:, 0].values,
            y=node_embedding_2d.iloc[:, 1].values,
            mode="markers",
            name="markers",
            text=list(node_embedding_2d.index),
            textposition="bottom center",
            marker=dict(
              size=6,
              color=colors[0],
              opacity=0.9,
              # colorscale="Rainbow",
              line_width=1
            ),
        ),
        row=math.ceil((i + 1) / cols), col=(i % cols) + 1
    )
  fig.update_layout(height=500*rows, width=1000, 
                    title_text="Embeddings", showlegend=True)
  fig.show()

def get_random_walk(graph, node, n_steps=4):
  random.seed(1)
  # Given a graph and a node, return a random walk starting from the node   
  local_path = [str(node),]
  target_node = node  
  for _ in range(n_steps):
    neighbors = list(nx.all_neighbors(graph, target_node))
    target_node = random.choice(neighbors)
    local_path.append(str(target_node))
  return local_path

def info_graph(graph):
  """ print(f"Radius: {nx.radius(graph)}")
  print(f"Diameter: {nx.diameter(graph)}")
  print(f"Eccentricity: {nx.eccentricity(graph)}")
  print(f"Center: {nx.center(graph)}")
  print(f"Periphery: {nx.periphery(graph)}") """
  print(f"Density: {nx.density(graph)}")
  print(f"Length: {len(graph)}")
  print(f"Nodes: {sorted(graph.nodes())}")
  print(f"N° nodes: {graph.number_of_nodes()}")
  print(f"Edges: {graph.edges()}")
  print(f"N° edges: {graph.number_of_edges()}")

def join_sub(df, blocks):
  # blocks= [(start1, end1), (start2, end2), ...]

  sdf = df.iloc[:, blocks[0][0]:blocks[0][1]]

  for block in blocks[1:]:
    sdf = sdf.join(df.iloc[:, block[0]:block[1]])
  return sdf

def transpose(df):
  df = df.T
  df.reset_index(drop=True, inplace=True)
  return df

def build_graph(matrix, threshold=0.5):
  edges = []
  for i in matrix.index:
    for j in matrix.columns:
      if i != j:
        if not math.isnan(matrix[i][j]) and abs(matrix[i][j]) >= threshold:
          edges.append([i, j])
  return edges

def build_graph_weight(matrix, threshold=0.5):
  edges = []
  for i in matrix.index:
    for j in matrix.columns:
      if i != j:
        if not math.isnan(matrix[i][j]) and abs(matrix[i][j]) >= threshold:
          edges.append([i, j, matrix[i][j]])
          """ if matrix[i][j] < 0:
            edges.append([i, j, 0.1])
          else:
            edges.append([i, j, matrix[i][j]]) """
  return edges

def deepwalk(G, num_walk, num_step):
  walk_paths = []
  for node in G.nodes():
    for _ in range(num_walk):
      walk_paths.append(get_random_walk(G, node, n_steps=num_step))
  return walk_paths

def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index2entity)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(10, 10))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

def detail(model):
  words = list(model.wv.vocab)
  print("Words\t", words)
  words_id = list(model.wv.index2word)
  print("Words id", words_id)
  words_id = list(model.wv.index2entity)
  print("Words id", words_id)
  print("Nodes\t", G.nodes())
  # print("Vector", model.wv["22"])
  print("Vector", model.wv.get_vector("1"))

def silhouette(X, k):
  # Generating the sample data from make_blobs
  # This particular setting has one distinct cluster and 3 clusters placed close
  # together.

  range_n_clusters = range(2, k)

  for n_clusters in range_n_clusters:
      # Create a subplot with 1 row and 2 columns
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
      # fig.set_size_inches(18, 7)

      # The 1st subplot is the silhouette plot
      # The silhouette coefficient can range from -1, 1 but in this example all
      # lie within [-0.1, 1]
      ax1.set_xlim([-0.1, 1])
      # The (n_clusters+1)*10 is for inserting blank space between silhouette
      # plots of individual clusters, to demarcate them clearly.
      ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

      # Initialize the clusterer with n_clusters value and a random generator
      # seed of 10 for reproducibility.
      clusterer = KMeans(n_clusters=n_clusters, random_state=10)
      cluster_labels = clusterer.fit_predict(X)

      # The silhouette_score gives the average value for all the samples.
      # This gives a perspective into the density and separation of the formed
      # clusters
      silhouette_avg = silhouette_score(X, cluster_labels)
      print(
          "For n_clusters =",
          n_clusters,
          "The average silhouette_score is :",
          silhouette_avg,
      )

      # Compute the silhouette scores for each sample
      sample_silhouette_values = silhouette_samples(X, cluster_labels)

      y_lower = 10
      for i in range(n_clusters):
          # Aggregate the silhouette scores for samples belonging to
          # cluster i, and sort them
          ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

          ith_cluster_silhouette_values.sort()

          size_cluster_i = ith_cluster_silhouette_values.shape[0]
          y_upper = y_lower + size_cluster_i

          color = cm.nipy_spectral(float(i) / n_clusters)
          ax1.fill_betweenx(
              np.arange(y_lower, y_upper),
              0,
              ith_cluster_silhouette_values,
              facecolor=color,
              edgecolor=color,
              alpha=0.7,
          )

          # Label the silhouette plots with their cluster numbers at the middle
          ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

          # Compute the new y_lower for next plot
          y_lower = y_upper + 10  # 10 for the 0 samples

      ax1.set_title("The silhouette plot for the various clusters.")
      ax1.set_xlabel("The silhouette coefficient values")
      ax1.set_ylabel("Cluster label")

      # The vertical line for average silhouette score of all the values
      ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

      ax1.set_yticks([])  # Clear the yaxis labels / ticks
      ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

      # 2nd Plot showing the actual clusters formed
      colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
      ax2.scatter(
          X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
      )

      # Labeling the clusters
      centers = clusterer.cluster_centers_
      # Draw white circles at cluster centers
      ax2.scatter(
          centers[:, 0],
          centers[:, 1],
          marker="o",
          c="white",
          alpha=1,
          s=200,
          edgecolor="k",
      )

      for i, c in enumerate(centers):
          ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

      ax2.set_title("The visualization of the clustered data.")
      ax2.set_xlabel("Feature space for the 1st feature")
      ax2.set_ylabel("Feature space for the 2nd feature")

      plt.suptitle(
          "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
          % n_clusters,
          fontsize=14,
          fontweight="bold",
      )

  plt.show()

def complete_nodes(df, length):
  indexes = df.index
  for index in range(length):
    if index not in indexes:
      df.loc[index] = [-1]
    
  df = df.sort_index()  # sorting by index
  return df

def matching():
  fig = make_subplots(rows=3, cols=2,
                      subplot_titles=("Raw data: 111-125", "Raw data: 411-425", 
                                      "Process data: 111-125", "Process data: 411-425"),
                      horizontal_spacing=0.05, vertical_spacing=0.05)

  fig.add_trace(
      go.Scatter(
          x=node_embeddings_2d1[:,0],
          y=node_embeddings_2d1[:,1],
          mode="markers",
          text=labels1,
          textposition="bottom center",
          marker=dict(
            size=8,
            color=labels1,
            opacity=0.9,
            colorscale="Rainbow",
            line_width=0.5
          ),
      ),
      row=1, col=1
  )

  fig.add_trace(
      go.Scatter(
          x=node_embeddings_2d2[:,0],
          y=node_embeddings_2d2[:,1],
          mode="markers",
          text=labels2,
          textposition="bottom center",
          marker=dict(
            size=8,
            color=labels2,
            opacity=0.9,
            colorscale="Rainbow",
            line_width=0.5
          ),
      ),
      row=1, col=2
  )

  fig.add_trace(
      go.Scatter(
          x=node_embeddings_2d3[:,0],
          y=node_embeddings_2d3[:,1],
          mode="markers+text",
          text=node_ids3,
          textposition="bottom center",
          marker=dict(
            size=8,
            color=labels3,
            opacity=0.9,
            colorscale="Rainbow", # ["red", "blue"],
            line_width=0.5
          ),
      ),
      row=2, col=1
  )

  fig.add_trace(
      go.Scatter(
          x=node_embeddings_2d4[:,0],
          y=node_embeddings_2d4[:,1],
          mode="markers+text",
          text=node_ids4,
          textposition="bottom center",
          marker=dict(
            size=8,
            color=labels4,
            opacity=0.9,
            colorscale="Rainbow", # ["red", "blue"],
            line_width=0.5
          ),
      ),
      row=2, col=2
  )

  labels1_ = [2] * len(node_ids1)
  size1_ = [8] * len(node_ids1)
  opacity1_ = [0.1] * len(node_ids1)
  for k, node_id in enumerate(node_ids3):
    if node_id in node_ids1:
      index = node_ids1.index(node_id)
      labels1_[index] = labels3[k]
      # size1_[index] = 10
      opacity1_[index] = 0.9

  fig.add_trace(
      go.Scatter(
          x=node_embeddings_2d1[:,0],
          y=node_embeddings_2d1[:,1],
          mode="markers",
          text=node_ids1,
          textposition="bottom center",
          marker=dict(
            size=8, # size1_,
            color=labels1_,
            opacity=opacity1_,
            colorscale="Rainbow", # ["red", "blue", "gray"],
            line_width=0.5
          ),
      ),
      row=3, col=1
  )

  labels2_ = [2] * len(node_ids2)
  size2_ = [8] * len(node_ids2)
  opacity2_ = [0.1] * len(node_ids2)
  for k, node_id in enumerate(node_ids4):
    if node_id in node_ids2:
      index = node_ids2.index(node_id)
      labels2_[index] = labels4[k]
      # size2_[index] = 10
      opacity2_[index] = 0.9

  fig.add_trace(
      go.Scatter(
          x=node_embeddings_2d2[:,0],
          y=node_embeddings_2d2[:,1],
          mode="markers",
          text=node_ids2,
          textposition="bottom center",
          marker=dict(
            size=8, # size2_,
            color=labels2_,
            opacity=opacity2_,
            colorscale="Rainbow", # ["red", "blue", "gray"],
            line_width=0.5
          ),
      ),
      row=3, col=2
  )

  fig.update_layout(height=1000, width=1000, title_text="Clustering Embeddings",
                    showlegend=False)
  fig.show()

  """ fig = px.scatter(df_plot1, x="component1", y="component2", hover_data=["node_id"], 
                  color="color", text="node_id")
  fig.update_layout(
      height=800,
      title_text='GDP and Life Expectancy (Americas, 2007)'
  )
  HTML(fig.to_html()) """

def similarity(df_embedding, w1, w2):
  u = df_embedding.loc[w1].to_numpy()
  v = df_embedding.loc[w2].to_numpy()

  similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
  return similarity
# similarity(node_embeddings1, '1', '173')

# model1.wv["798"]
def most_similar(df_embedding, w, topn=10):
  # u = embedding.loc[w].to_numpy() # model1.wv["1"]
  similarities = []
  for index in df_embedding.index:
    similar = similarity(df_embedding, w, index)
    similarities.append((index, similar))
  similarities = np.array(similarities)

  sorted_array = similarities[np.argsort(similarities[:, 1])]
  sorted_array = sorted_array[::-1][1:topn + 1]
  return sorted_array
# most_similar(node_embeddings1, "1", 10)

# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist

def similarity_cos(u, v):
  similarity = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
  return similarity

def euclidean_distance(u, v):
  dist = np.linalg.norm(u - v)
  return dist

def most_similar2(embedding1, embedding2, topn=10, metric="euclidean"):
  # u = embedding.loc[w].to_numpy() # model1.wv["1"]
  data = []
  for w1 in embedding1.index:
    u = embedding1.loc[w1].to_numpy()
    similarities = []
    for w2 in embedding2.index:
      v = embedding2.loc[w2].to_numpy()
      # similar = euclidean_distance(u, v)
      # similar = similarity_cos(u, v)
      similar = distance.cdist([u], [v], metric=metric)
      similarities.append((int(w1), int(w2), similar[0][0]))

    similarities = np.array(similarities)

    sorted_array = similarities[np.argsort(similarities[:, 2])]
    if metric == "cosine":
      sorted_array = sorted_array[:][:topn]
    else:
      sorted_array = sorted_array[:topn]
    
    for item in sorted_array:
      data.append(item)
  df = pd.DataFrame(data, columns=["u", "u'", "{}".format(metric)])
  df["u"] = df["u"].astype("int")
  df["u'"] = df["u'"].astype("int")

  return df

def get_epsilon(X_train):
  neigh = NearestNeighbors(n_neighbors=2 * X_train.shape[1])
  nbrs = neigh.fit(X_train)
  distances, indices = nbrs.kneighbors(X_train)

  # Plotting K-distance Graph
  distances = np.sort(distances, axis=0)
  distances_ = distances[:,1]

  i = np.arange(len(distances))
  knee = KneeLocator(i, distances_, S=1, curve='convex', direction='increasing', interp_method='polynomial')

  plt.figure(figsize=(12, 6))
  knee.plot_knee()
  plt.xlabel("Points")
  plt.ylabel("Distance")
  plt.grid()

  print(distances[knee.knee])