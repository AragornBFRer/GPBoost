import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate fixed effects F(x)
def generate_fixed_effects(n, d=9, C=None):
    """
    Generate fixed effects F(x) for n samples.
    
    Parameters:
        n (int): Number of samples.
        d (int): Number of predictor variables (default=9).
        C (float): Scaling constant to adjust variance of F(x) to 1.
    
    Returns:
        F (numpy.ndarray): Fixed effect values of shape (n,).
        X (numpy.ndarray): Predictor variables of shape (n, d).
    """
    X = np.random.normal(0, 1, size=(n, d))  # Generate X ~ N(0, I_d)
    # Define F(x) as given in the problem
    Fx = 2 * X[:, 0] + X[:, 1]**2 + 4 * (X[:, 2] > 0) + 2 * np.log(np.abs(X[:, 0] * X[:, 2]))
    
    if C is None:
        # Calculate C such that var(F(x)) = 1
        C = 1 / np.std(Fx)
    
    F = C * Fx
    return F, X

# Function to generate grouped random effects
def generate_grouped_random_effects(n, m, sigma1_squared=1):
    """
    Generate grouped random effects Zb.

    Parameters:
        n (int): Total number of samples.
        m (int): Number of groups.
        sigma1_squared (float): Variance of random effects.

    Returns:
        Zb (numpy.ndarray): Random effects of shape (n,).
        group_ids (numpy.ndarray): Group IDs for each sample.
    """
    group_ids = np.random.choice(m, size=n)  # Assign each sample to a group
    b = np.random.normal(0, np.sqrt(sigma1_squared), size=m)  # Random effects for each group
    Zb = b[group_ids]  # Assign random effects to each sample
    return Zb, group_ids

# Function to generate spatial Gaussian process random effects
def generate_spatial_gp(n, locations, sigma1_squared=1, rho=0.1):
    """
    Generate spatial Gaussian process random effects Zb.

    Parameters:
        n (int): Number of samples.
        locations (numpy.ndarray): Locations of shape (n, 2).
        sigma1_squared (float): Variance of the Gaussian process.
        rho (float): Range parameter for the exponential covariance function.

    Returns:
        Zb (numpy.ndarray): Spatial random effects of shape (n,).
    """
    # Exponential covariance function
    dists = cdist(locations, locations)
    covariance_matrix = sigma1_squared * np.exp(-dists / rho)
    
    # Simulate random effects from multivariate normal distribution
    Zb = multivariate_normal.rvs(mean=np.zeros(n), cov=covariance_matrix)
    return Zb

# Function to generate data based on the mixed effects model
def generate_data(n, model_type="grouped", m=500, sigma1_squared=1, sigma_squared=1, rho=0.1):
    """
    Generate data based on the mixed effects model.

    Parameters:
        n (int): Number of samples.
        model_type (str): Type of random effects model ("grouped" or "spatial").
        m (int): Number of groups (only for grouped random effects).
        sigma1_squared (float): Variance of random effects.
        sigma_squared (float): Variance of error term.
        rho (float): Range parameter for spatial GP (only for spatial model).

    Returns:
        y (numpy.ndarray): Response variable of shape (n,).
        X (numpy.ndarray): Predictor variables of shape (n, 9).
        Zb (numpy.ndarray): Random effects of shape (n,).
        epsilon (numpy.ndarray): Error term of shape (n,).
        group_ids or locations (numpy.ndarray): Group IDs or spatial locations.
    """
    # Generate fixed effects
    F, X = generate_fixed_effects(n)
    
    if model_type == "grouped":
        # Generate grouped random effects
        Zb, group_ids = generate_grouped_random_effects(n, m, sigma1_squared)
        locations_or_groups = group_ids
    elif model_type == "spatial":
        # Generate spatial locations
        locations = np.random.uniform(0, 1, size=(n, 2))
        # Generate spatial random effects
        Zb = generate_spatial_gp(n, locations, sigma1_squared, rho)
        locations_or_groups = locations
    else:
        raise ValueError("Invalid model_type. Choose from 'grouped' or 'spatial'.")
    
    # Generate error term
    epsilon = np.random.normal(0, np.sqrt(sigma_squared), size=n)
    
    # Combine to generate response variable y
    y = F + Zb + epsilon
    
    return y, X, Zb, epsilon, locations_or_groups

# Function to generate training and test datasets
def generate_train_test_data(n_train, n_test, model_type, m=500, sigma1_squared=1, sigma_squared=1, rho=0.1):
    """
    Generate training and test datasets (interpolation and extrapolation).

    Parameters:
        n_train (int): Number of training samples.
        n_test (int): Number of test samples.
        model_type (str): Type of random effects model ("grouped" or "spatial").
        m (int): Number of groups (only for grouped random effects).
        sigma1_squared (float): Variance of random effects.
        sigma_squared (float): Variance of error term.
        rho (float): Range parameter for spatial GP (only for spatial model).

    Returns:
        train_data (dict): Training data.
        test_data_interpolation (dict): Interpolation test data.
        test_data_extrapolation (dict): Extrapolation test data.
    """
    # Generate training data
    train_data = generate_data(n_train, model_type, m, sigma1_squared, sigma_squared, rho)
    
    # Generate test data for interpolation and extrapolation
    if model_type == "grouped":
        # Interpolation: Use the same groups as in training
        test_data_interpolation = generate_data(n_test, model_type, m, sigma1_squared, sigma_squared, rho)
        # Extrapolation: Generate new groups
        test_data_extrapolation = generate_data(n_test, model_type, m, sigma1_squared, sigma_squared, rho)
    elif model_type == "spatial":
        # Interpolation: Sample locations from [0, 0.5] x [0, 0.5]
        interpolation_locations = np.random.uniform(0, 0.5, size=(n_test, 2))
        Zb_interpolation = generate_spatial_gp(n_test, interpolation_locations, sigma1_squared, rho)
        F_interpolation, X_interpolation = generate_fixed_effects(n_test)
        epsilon_interpolation = np.random.normal(0, np.sqrt(sigma_squared), size=n_test)
        y_interpolation = F_interpolation + Zb_interpolation + epsilon_interpolation
        test_data_interpolation = {
            "y": y_interpolation,
            "X": X_interpolation,
            "Zb": Zb_interpolation,
            "epsilon": epsilon_interpolation,
            "locations": interpolation_locations,
        }
        
        # Extrapolation: Sample locations from [0.5, 1] x [0.5, 1]
        extrapolation_locations = np.random.uniform(0.5, 1, size=(n_test, 2))
        Zb_extrapolation = generate_spatial_gp(n_test, extrapolation_locations, sigma1_squared, rho)
        F_extrapolation, X_extrapolation = generate_fixed_effects(n_test)
        epsilon_extrapolation = np.random.normal(0, np.sqrt(sigma_squared), size=n_test)
        y_extrapolation = F_extrapolation + Zb_extrapolation + epsilon_extrapolation
        test_data_extrapolation = {
            "y": y_extrapolation,
            "X": X_extrapolation,
            "Zb": Zb_extrapolation,
            "epsilon": epsilon_extrapolation,
            "locations": extrapolation_locations,
        }
    else:
        raise ValueError("Invalid model_type. Choose from 'grouped' or 'spatial'.")
    
    return train_data, test_data_interpolation, test_data_extrapolation

# Example usage
n_train = 5000  # Training sample size
n_test = 500  # Test sample size

# Grouped random effects model
train_grouped, test_interp_grouped, test_extrap_grouped = generate_train_test_data(
    n_train, n_test, model_type="grouped", m=500, sigma1_squared=1, sigma_squared=1
)

# Spatial Gaussian process model
train_spatial, test_interp_spatial, test_extrap_spatial = generate_train_test_data(
    n_train, n_test, model_type="spatial", sigma1_squared=1, sigma_squared=1, rho=0.1
)


import matplotlib.pyplot as plt

# Function to visualize the data
def visualize_data(train_data, test_data_interpolation, test_data_extrapolation, model_type):
    y_train, X_train, Zb_train, epsilon_train, locations_or_groups_train = train_data
    y_test_interp, X_test_interp, Zb_test_interp, epsilon_test_interp, locations_or_groups_test_interp = test_data_interpolation
    y_test_extrap, X_test_extrap, Zb_test_extrap, epsilon_test_extrap, locations_or_groups_test_extrap = test_data_extrapolation
    
    if model_type == "grouped":
        # Visualize grouped random effects
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(Zb_train, bins=30, alpha=0.7, label='Train')
        plt.hist(Zb_test_interp, bins=30, alpha=0.7, label='Test Interpolation')
        plt.hist(Zb_test_extrap, bins=30, alpha=0.7, label='Test Extrapolation')
        plt.xlabel('Grouped Random Effects')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Distribution of Grouped Random Effects')
        
        # Visualize response variable y
        plt.subplot(1, 2, 2)
        plt.hist(y_train, bins=30, alpha=0.7, label='Train')
        plt.hist(y_test_interp, bins=30, alpha=0.7, label='Test Interpolation')
        plt.hist(y_test_extrap, bins=30, alpha=0.7, label='Test Extrapolation')
        plt.xlabel('Response Variable y')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Distribution of Response Variable y')
        
    elif model_type == "spatial":
        # Visualize spatial random effects
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(locations_or_groups_train[:, 0], locations_or_groups_train[:, 1], c=Zb_train, cmap='viridis', label='Train', alpha=0.7)
        plt.colorbar(label='Spatial Random Effects')
        plt.xlabel('Location X1')
        plt.ylabel('Location X2')
        plt.legend()
        plt.title('Spatial Random Effects (Train)')
        
        # Visualize response variable y
        plt.subplot(1, 2, 2)
        plt.scatter(locations_or_groups_train[:, 0], locations_or_groups_train[:, 1], c=y_train, cmap='viridis', label='Train', alpha=0.7)
        plt.colorbar(label='Response Variable y')
        plt.xlabel('Location X1')
        plt.ylabel('Location X2')
        plt.legend()
        plt.title('Response Variable y (Train)')
        
    else:
        raise ValueError("Invalid model_type. Choose from 'grouped' or 'spatial'.")
    
    # Show plots
    plt.tight_layout()
    plt.show()

# Visualize the grouped random effects model data
visualize_data(train_grouped, test_interp_grouped, test_extrap_grouped, model_type="grouped")

# Visualize the spatial Gaussian process model data
visualize_data(train_spatial, test_interp_spatial, test_extrap_spatial, model_type="spatial")
