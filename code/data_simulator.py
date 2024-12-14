import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

np.random.seed(42)

def generate_fixed_effects(n, d=9, C=None):
    X = np.random.normal(0, 1, size=(n, d))
    Fx = 2 * X[:, 0] + X[:, 1]**2 + 4 * (X[:, 2] > 0) + 2 * np.log(np.abs(X[:, 0] * X[:, 2]))
    
    if C is None:
        C = 1 / np.std(Fx)
    
    F = C * Fx
    return F, X

def generate_grouped_random_effects(n, m, sigma1_squared=1):
    group_ids = np.random.choice(m, size=n)
    b = np.random.normal(0, np.sqrt(sigma1_squared), size=m)
    Zb = b[group_ids]
    return Zb, group_ids

def generate_spatial_gp(n, locations, sigma1_squared=1, rho=0.1):
    dists = cdist(locations, locations)
    covariance_matrix = sigma1_squared * np.exp(-dists / rho)
    
    Zb = multivariate_normal.rvs(mean=np.zeros(n), cov=covariance_matrix)
    return Zb

def generate_data(n, model_type="grouped", m=500, sigma1_squared=1, sigma_squared=1, rho=0.1, locations=None):
    F, X = generate_fixed_effects(n)
    if model_type == "grouped":
        Zb, group_ids = generate_grouped_random_effects(n, m, sigma1_squared)
        locations_or_groups = group_ids
    elif model_type == "spatial":
        if locations is None:
            raise ValueError("For spatial model, locations must be provided.")
        Zb = generate_spatial_gp(n, locations, sigma1_squared, rho)
        locations_or_groups = locations
    else:
        raise ValueError("Invalid model_type. Choose from 'grouped' or 'spatial'.")
    epsilon = np.random.normal(0, np.sqrt(sigma_squared), size=n)
    y = F + Zb + epsilon
    
    return y, X, Zb, epsilon, locations_or_groups

def generate_locations(n, area="train"):
    if area == "train" or area == "interpolation":
        # [0, 1]^2 \ [0.5, 1]^2
        locations = []
        while len(locations) < n:
            loc = np.random.uniform(0, 1, size=(n, 2))
            mask = (loc[:, 0] < 0.5) | (loc[:, 1] < 0.5)  # Exclude [0.5, 1]^2
            locations.extend(loc[mask])
        locations = np.array(locations[:n])
    elif area == "extrapolation":
        # [0.5, 1]^2
        locations = np.random.uniform(0.5, 1, size=(n, 2))
    else:
        raise ValueError("Invalid area. Choose from 'train', 'interpolation', or 'extrapolation'.")
    
    return locations

def generate_train_test_data(n_train, n_test, model_type, m=500, sigma1_squared=1, sigma_squared=1, rho=0.1):
    if model_type == "spatial":
        train_locations = generate_locations(n_train, area="train")
        interp_locations = generate_locations(n_test, area="interpolation")
        extrap_locations = generate_locations(n_test, area="extrapolation")
        
        train_data = generate_data(n_train, model_type, 
                                   sigma1_squared=sigma1_squared, 
                                   sigma_squared=sigma_squared, rho=rho, 
                                   locations=train_locations)
        test_data_interpolation = generate_data(n_test, model_type, 
                                                sigma1_squared=sigma1_squared, 
                                                sigma_squared=sigma_squared, 
                                                rho=rho, locations=interp_locations)
        test_data_extrapolation = generate_data(n_test, model_type, 
                                                sigma1_squared=sigma1_squared, 
                                                sigma_squared=sigma_squared, 
                                                rho=rho, locations=extrap_locations)
    else:
        train_data = generate_data(n_train, model_type, m=m, 
                                   sigma1_squared=sigma1_squared, 
                                   sigma_squared=sigma_squared)
        test_data_interpolation = generate_data(n_test, model_type, m=m, 
                                                sigma1_squared=sigma1_squared, 
                                                sigma_squared=sigma_squared)
        test_data_extrapolation = generate_data(n_test, model_type, m=m, 
                                                sigma1_squared=sigma1_squared, 
                                                sigma_squared=sigma_squared)
    
    return train_data, test_data_interpolation, test_data_extrapolation

n_train = 500
n_test = 500

train_spatial, test_interp_spatial, test_extrap_spatial = generate_train_test_data(
    n_train, n_test, model_type="spatial", sigma1_squared=1, sigma_squared=1, rho=0.1
)
