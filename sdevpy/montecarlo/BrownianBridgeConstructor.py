

def brownian_bridge(n_paths, n_steps, dim, T, sobol_engine):
    
    dt = T / n_steps
    
    # total Sobol dimension required
    total_dim = dim * n_steps
    
    U = sobol_engine.random(n_paths)
    
    # convert to normals
    Z = norm.ppf(U)
    
    # reshape
    Z = Z.reshape(n_paths, dim, n_steps)
    
    W = np.zeros((n_paths, dim, n_steps+1))
    
    for d in range(dim):
        
        # first assign W(T)
        W[:, d, -1] = np.sqrt(T) * Z[:, d, 0]
        
        time_grid = np.linspace(0, T, n_steps+1)
        
        # recursive midpoint fill
        intervals = [(0, n_steps)]
        
        z_index = 1
        
        while intervals:
            
            start, end = intervals.pop(0)
            mid = (start + end) // 2
            
            if mid == start or mid == end:
                continue
            
            t_start = time_grid[start]
            t_mid = time_grid[mid]
            t_end = time_grid[end]
            
            mean = ((t_mid - t_start) * W[:, d, end] +
                    (t_end - t_mid) * W[:, d, start]) / (t_end - t_start)
            
            var = (t_mid - t_start) * (t_end - t_mid) / (t_end - t_start)
            
            W[:, d, mid] = mean + np.sqrt(var) * Z[:, d, z_index]
            
            z_index += 1
            
            intervals.append((start, mid))
            intervals.append((mid, end))
    
    # convert to increments
    dW = np.diff(W, axis=2)
    
    return dW
