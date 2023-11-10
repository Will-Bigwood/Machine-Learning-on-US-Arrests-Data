# IMPORT LIBRARIES:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import geopandas as gpd


# DEFINE FUNCTIONS:

def create_corr_heatmap(df):
  # get correlations
  corrs = df.corr()

  # pick appropriate palette for heatmap
  colormap = sns.diverging_palette(220, 10, as_cmap=True)  
  
  # Plot figsize
  fig, ax = plt.subplots(figsize=(15, 15))

  # to mask mirrored side of heatmap
  dropSelf = np.zeros_like(corrs)
  dropSelf[np.triu_indices_from(dropSelf)] = True

  # Generate Heat Map, with annotations of the correlation strength and 2-decimal floats
  ax = sns.heatmap(corrs, cmap=colormap, annot=True, fmt=".2f", mask=dropSelf, linewidth=2)

  plt.show()


def create_variable_hist(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Histograms of Numerical Variables")

    for i, column in enumerate(df.columns):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        ax.hist(df[column], bins=10, color='skyblue', edgecolor='black')
        ax.set_title(f'Histogram of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def create_pair_plot(df):
    sns.pairplot(df, diag_kind="kde", markers="o")
    plt.show()


def create_murder_scatters(df, indices):
    for idx in indices:
        column = df.columns[idx]
        plt.scatter(df[column], df['Murder'])
        plt.xlabel(column)
        plt.ylabel('Murder')
        plt.show()


def generate_data_stats(df):
    # Calculate statistics and the number of missing values in each column
    statistics = df.describe()
    missing_values = df.isnull().sum()

    print("Descriptive Statistics:")
    print(f"{statistics}\n")
    print("Missing Values:")
    print(f"{missing_values}\n")

    # Create a Histogram to show the distribution for each numerical variable in the dataset
    create_variable_hist(df)

    # Create a correlation heatmap to show the correlations between the numerical variables in the dataset
    create_corr_heatmap(df)

    # Create a pair plot showing the relationship between each pair of variables
    create_pair_plot(df)

    # Create two scatter plots showing ('Assault' vs 'Murder') and ('Rape' vs 'Murder')
    create_murder_scatters(df, [1, 3])


def create_scree_plot(cumulative_explained_variance):
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid()
    plt.show()


def create_biplot(pca_df, df):
    # Create a scatter plot of the principal components
    plt.figure(figsize=(8, 8))
    plt.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Add arrows representing variable loadings
    for i, (variance, component) in enumerate(zip(pca.explained_variance_, pca.components_)):
        pc1_load = component[0]
        pc2_load = component[1]
        plt.arrow(0, 0, pc1_load * np.sqrt(variance), pc2_load * np.sqrt(variance), color='r', alpha=0.5)
        plt.text(pc1_load * np.sqrt(variance), pc2_load * np.sqrt(variance), df.columns[i], color='r')

    plt.grid()
    plt.show()


def create_pca_corr_map(df):
    # Plot a correlation heatmap for the principal components
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(df.corr(), cmap=colormap,linewidth=1)
    plt.show()


def generate_pca_stats(pca_df):

    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance) / np.sum(explained_variance)

    # Print statistics for each principal component
    for i, (variance, ratio, cumulative_ratio) in enumerate(zip(explained_variance, explained_variance_ratio, cumulative_explained_variance)):
        print(f"Principal Component {i + 1}:")
        print(f"Standard Deviation (StdDev): {np.sqrt(variance)}")
        print(f"Proportion of Variance Explained: {ratio:.4f}")
        print(f"Cumulative Proportion Explained: {cumulative_ratio:.4f}")
        print()

    # Plot a biplot to visualise the data in the principal componants
    create_biplot(pca_df, df)

    # We can then look at a Cumulative Explained Variance plot to see how much of the data is explained by each principal componant.
    create_scree_plot(cumulative_explained_variance)


def perform_pca(df, X):
    principal_components = pca.fit_transform(X)
    pca_df = pd.DataFrame(principal_components, index=df.index)
    return pca_df


def eval_Kmeans(x, k, r):
    kmeans = KMeans(n_clusters=k, random_state=r)
    kmeans.fit(x)    
    return kmeans.inertia_


def elbow_Kmeans(x, max_k=10, r=123):
    within_cluster_vars = [eval_Kmeans(x, k, r) for k in range(1, max_k+1)]
    plt.plot(range(1, 11), within_cluster_vars,marker='o')
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.show()


def scatter_Kmeans(x, k, r=123):
    # Create a K-Means model with the specified number of clusters (k)
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=r)
    
    # Fit the K-Means model and predict cluster labels
    y_pred = kmeans.fit_predict(x)
    
    # Define colors for the clusters
    colours = 'rbgcmy'
    
    # Convert the DataFrame to a NumPy array for plotting
    x_array = x.values
    
    # Plot the data points and cluster centers for each cluster
    for c in range(k):
        plt.scatter(x_array[y_pred == c, 0], x_array[y_pred == c, 1], c=colours[c], label='Cluster {}'.format(c))
        plt.scatter(kmeans.cluster_centers_[c, 0], kmeans.cluster_centers_[c, 1], marker='x', c='black')

    # Calculate the Silhouette Score and display it in the plot
    score = round(silhouette_score(x_array, kmeans.labels_, metric='euclidean'), 2)
    title = f'K-Means Clustering (k={k}), silhouette={score}'
    plt.title(title, loc='right', fontdict={'fontsize': 16}, pad=-14)
    
    
    # Customize the legend
    plt.legend(title='Clusters', loc='upper right')
    plt.show()


def perform_kmeans_clustering(data, n_clusters, rseed):
    # Initialize K-Means with a specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=rseed, max_iter=500)

    # Fit K-Means to the data
    kmeans.fit(data)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Calculate the Silhouette Score
    silhouette_avg = silhouette_score(data, cluster_labels, metric='euclidean')

    # Print cluster counts and silhouette score
    cluster_counts = pd.Series(cluster_labels).value_counts()
    print(f"Cluster Counts: {cluster_counts}")
    print(f"Silhouette Score for K={n_clusters}: {silhouette_avg}")

    return kmeans


def visualise_cluster_scatterplots(df, x_variable, y_variables, cluster_col_name, palette='Set1'):
    for y_variable in y_variables:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=x_variable, y=y_variable, data=df, hue=cluster_col_name, palette=palette)
        plt.title(f'{x_variable} vs {y_variable}')

        for i, state in enumerate(df.index):
            plt.text(df[x_variable][i], df[y_variable][i], state, fontsize=8, alpha=0.7)

        plt.show()


def create_k_means_cluster_pair_plot(df, cluster_col_name, palette='Set1'):
    sns.pairplot(data=df, hue=cluster_col_name, palette=palette, diag_kind='kde', markers="o", plot_kws={'alpha': 0.7})
    plt.show()


def create_hierarchical_cluster_pair_plot(df, cluster_col_name, palette='Set1'):
    # Create a pair plot with hue based on the cluster_col_name
    sns.pairplot(data=df, hue=cluster_col_name, palette=palette, diag_kind='kde', markers="o", plot_kws={'alpha': 0.7})
    plt.show()


def work_out_linkage(X):

    linkage_types = ['complete', 'average', 'single']

    plt.figure(figsize=(15, 5))
    for i, method in enumerate(linkage_types):
        plt.subplot(1, 3, i + 1)
        plt.title(f"{method.capitalize()} Linkage Dendrogram")
        dend = dendrogram(linkage(X, method=method))
    plt.show()


def work_out_K(X, selected_linkage):

    plt.title(f"{selected_linkage.capitalize()} Linkage Dendrogram")
    dend = dendrogram(linkage(X, method=selected_linkage))
    plt.show()


def create_hierarchical_cluster_model(df, pca_df, n_clusters, selected_linkage):
    
    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=selected_linkage)
    df['cluster_label'] = hierarchical_cluster.fit_predict(pca_df.values)

    return df


def print_cluster_info():
    # Calculate and print the counts of data points in each cluster
    cluster_counts = df['cluster_label'].value_counts()
    print("\nCluster Counts for Hierarchical Clustering:")
    print(cluster_counts)

    # Calculate the silhouette score
    silhouette_avg = silhouette_score(pca_df, df['cluster_label'], metric='euclidean')
    print("\nSilhouette Score for Hierarchical Clustering:", silhouette_avg)

def match_gdf_to_df(df, gdf):
    # Convert the 'NAME' column in the geodataframe to lowercase and sort it alphabetically
    gdf['NAME'] = gdf['NAME'].str.lower()
    gdf = gdf.sort_values(by='NAME')

    # Convert the 'State' column in df to lowercase to match gdf
    df.index = df.index.str.lower()

    # Find common state names in both DataFrames
    common_states = df.index.intersection(gdf['NAME'])

    # Filter gdf to keep only rows with state names found in df
    gdf = gdf[gdf['NAME'].isin(common_states)]

def create_us_state_map(merged, clustering_technique):
    # Create a map
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    merged.plot(column='cluster_label', cmap='Set1', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

    # Set the title and legend label
    ax.set_title(f'Violent Crime Rates for US states, Based on {clustering_technique} clustering', fontsize=16)
    ax.set_aspect('equal')
    # Set the x-axis limits to only show the left side
    ax.set_xlim(left=-180, right=-60)

    # Display the plot
    plt.show()

# MAIN SECTION:

# DATA PREPROCESSING AND EXPLORATION:

# Read files into relevant dataframes
df = pd.read_csv("UsArrests.csv")
gdf = gpd.read_file('tl_2021_us_state.shp')

# Change the 'City' column name to 'State' and then set that column as the index of the dataframe
df.rename(columns={'City': 'State'}, inplace=True)
df = df.set_index('State')


# Generate statistics and visalisations for exploratory analysis
generate_data_stats(df)


# PCA ON UNSTANDARDISED DATA:

# Perform regular PCA, without specifying a number of components, on the unstandardised data
pca = PCA(n_components=None)
X_unstandardised = df.values
pca_df = perform_pca(df, X_unstandardised)

# Generate statistics on the PCA
generate_pca_stats(pca_df)


# PCA ON STANDARDISED DATA:

# Scale the data and perform regular PCA, without specifying a number of components, on the standardised data
X_standardised = StandardScaler().fit_transform(df.values)
pca_df = perform_pca(df, X_standardised)

# Generate statistics on the PCA
generate_pca_stats(pca_df)


# APPLY PCA WITH 3 COMPONENTS:

# Perform PCA, specifying 3 components, on the standardised data
pca = PCA(n_components=3)
pca_df = perform_pca(df, X_standardised)


# Create a correlation heatmap for the principal components
create_pca_corr_map(pca_df)


# K-MEANS CLUSTERING:

rseed = 42

# Elbow method analysis
elbow_Kmeans(pca_df.values)

# Try K values from 2 to 4
for k in range(2, 5):
    scatter_Kmeans(pca_df, k, r=0)


# Perform K means clustering with 2 clusters on the pca dataset
kmeans = perform_kmeans_clustering(pca_df, n_clusters=2, rseed=rseed)

# Add the predicted cluster label column to the original dataframe
df['cluster_label'] = kmeans.predict(pca_df)


# Visualisation of clusters, with 'State' name labels, for Murder vs Assault, and Murder vs Rape.
y_variables = ['Assault', 'Rape']
visualise_cluster_scatterplots(df, 'Murder', y_variables, 'cluster_label')

# Visualisation of clusters on a pair-plot
create_k_means_cluster_pair_plot(df, 'cluster_label')

# Reset the dataframe to its original form for the next analysis
df = df.drop('cluster_label', axis=1)


# HIERARCHICAL CLUSTERING:

# Prepare the data
X = df.values

# Determin which linkage type to use
work_out_linkage(X)

# Select linkage type
L = 'complete'

# Work out the best number of clusters
work_out_K(X, L)

# Select a number of clusters
K = 4

# Create a hierarchical model, fit it to the PC data, and store the cluster labels in the original dataframe
df = create_hierarchical_cluster_model(df, pca_df, K, L)


# Create a pairplot for the numerical variables showing the different clusters
create_hierarchical_cluster_pair_plot(df, 'cluster_label')

# Create a scatter plot for Murder vs Assault showing which states are in which cluster
y_variables = ['Assault']
visualise_cluster_scatterplots(df, 'Murder', y_variables, 'cluster_label')

# Print the counts of data points in each cluster and the silhouette score
print_cluster_info()


# Define a mapping for renaming the hierarchical cluster labels based on earlier analysis
label_mapping = {
    0: 'Mid/Low',
    1: 'High',
    2: 'Mid/High',
    3: 'Low'
}

# Rename the 'cluster_label' column data based on the mapping
df['cluster_label'] = df['cluster_label'].replace(label_mapping)

# Inspect the US State shapefile geodataframe
print(f'Shapefile data:\n{gdf.head()}\n')
print(f'Shapefile data (shape):\n{gdf.shape}\n')
print(f'Shapefile Column Names:')
print(gdf.columns)

# There appear to be 55 states in the data, where as our dataset has only 50

# Inspect the state names
print(gdf['NAME'])


# Match the 'state' columns so that there are only 50 states in each dataframe
match_gdf_to_df(df, gdf)

# Merge GeoDataFrame and DataFrame based on the common state names
merged = gdf.merge(df, left_on='NAME', right_index=True)

# Create a map showing which cluster each state is in.
create_us_state_map(merged, 'Hierarchical')



df['cluster_label'] = kmeans.predict(pca_df)

# Define a mapping for renaming the hierarchical cluster labels based on earlier analysis
label_mapping = {
    0: 'Low',
    1: 'High'
}

# Rename the 'cluster_label' column based on the mapping
df['cluster_label'] = df['cluster_label'].replace(label_mapping)

# Merge GeoDataFrame and DataFrame based on the common state names
merged = gdf.merge(df, left_on='NAME', right_index=True)

# Create a map showing which cluster each state is in.
create_us_state_map(merged, 'K-Means')