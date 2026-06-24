def pfof_original(ra, dec, cz, czerr, perpll, losll, Pth, H0=100., Om0=0.3, Ode0=0.7, printConf=True):
    """  
    -----
    Compute group membership from galaxies' equatorial  coordinates using a probabilitiy
    friends-of-friends (PFoF) algorithm, based on the method of Liu et al. 2008. PFoF is
    a variant of FoF (see `foftools.fast_fof`, Berlind+2006), which treats galaxies as Gaussian
    probability distributions, allowing group membership selection to account for the 
    redshift errors of photometric redshift measurements. 
    In this function, the linking length must be fixed.   
 
    Arguments:
        ra (iterable): list of right-ascesnsion coordinates of galaxies in decimal degrees.
        dec (iterable): list of declination coordinates of galaxies in decimal degrees.
        cz (iterable): line-of-sight recessional velocities of galaxies in km/s.
        czerr (iterable): errors on redshifts of galaxies in km/s.
        perpll (float): perpendicular linking length in Mpc. 
        losll (float): line-of-sight linking length in Mpc.
        Pth (float): Threshold probability from which to construct the group catalog. If None, the
            function will return a NxN matrix of friendship probabilities.
        printConf (bool, default True): bool indicating whether to print confirmation at the end.
    Returns:
        grpid (np.array): list containing unique group ID numbers for each target in the input coordinates.
                The list will have shape len(ra).
    -----
    """
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0) # this puts everything in "per h" units.
    c=SPEED_OF_LIGHT
    Ngalaxies = len(ra)
    ra = np.float32(ra)
    dec = np.float32(dec)
    zz = np.float32(cz/c)
    zzerr = np.float32(czerr/c)
    assert (len(ra)==len(dec) and len(dec)==len(cz)),"RA/Dec/cz arrays must equivalent length."

    phi = np.deg2rad(ra)
    theta = np.pi/2. - np.deg2rad(dec)
    transv_cmvgdist = (cosmo.comoving_transverse_distance(zz).value)
    los_cmvgdist = (cosmo.comoving_distance(zz).value)
    dc_upper = los_cmvgdist + losll
    dc_lower = los_cmvgdist - losll

    z_arr_interp = np.arange(0.00001,2*np.max(zz), np.min(zzerr)/3)
    print(z_arr_interp)
    z_dc_interp = interp1d(cosmo.comoving_distance(z_arr_interp).value, z_arr_interp, fill_value=0, bounds_error=False)
    VL_lower = np.float32(zz - z_dc_interp(dc_lower))
    VL_upper = np.float32(z_dc_interp(dc_upper) - zz)
    friendship = np.zeros((Ngalaxies, Ngalaxies),dtype=np.int8)
    del z_arr_interp

    # Compute on-sky perpendicular distance
    col_theta=theta[:,None]
    col_phi=phi[:,None]
    half_angle = np.arcsin((np.sin((col_theta-theta)/2.0)**2.0 + np.sin(col_theta)*np.sin(theta)*np.sin((col_phi-phi)/2.0)**2.0)**0.5)
    column_transv_cmvgdist = transv_cmvgdist[:, None]
    dperp = (column_transv_cmvgdist + transv_cmvgdist) * half_angle # In Mpc
    del half_angle, column_transv_cmvgdist, transv_cmvgdist, los_cmvgdist, phi, theta, ra, dec, col_phi, col_theta

    # Compute line-of-sight probabilities
    prob_dlos=np.zeros((Ngalaxies, Ngalaxies),dtype=np.float32)
    np.fill_diagonal(prob_dlos,1)
    
    i_idx, j_idx = np.triu_indices(Ngalaxies, k=1) 
    mask = dperp[i_idx, j_idx] <= perpll
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    zz_i, zz_j = zz[i_idx], zz[j_idx]
    zzerr_i, zzerr_j = zzerr[i_idx], zzerr[j_idx]
    VL_lower_i, VL_upper_i = VL_lower[i_idx], VL_upper[i_idx]

    zmin_, zmax_ = np.min(zz)-3*np.max(zzerr),np.max(zz)+3*np.max(zzerr)
    meshZ = np.arange((zmin_ if zmin_>0 else 0), zmax_, np.percentile(zzerr, 1)/3, dtype=np.float32) # resolution adapts to dataset
   
    mesh_length = len(meshZ) # batch over meshZ instead of ngal.
    if mesh_length > 2000:
        vals = np.zeros(len(zz_i), dtype=np.float64)
        batch_idx = np.array_split(np.arange(mesh_length), int(mesh_length/300))
        print(f'Large redshift mesh (n={mesh_length}). Running PFoF in batches.')
        for idx in tqdm(batch_idx):
            vals += numeric_integration_pfof_vectorized(meshZ[idx], zz_i, zzerr_i, zz_j, zzerr_j, VL_lower_i, VL_upper_i)
    else:
        vals = numeric_integration_pfof_vectorized(meshZ, zz_i, zzerr_i, zz_j, zzerr_j, VL_lower_i, VL_upper_i)
    prob_dlos[i_idx, j_idx] = vals 
    prob_dlos[j_idx, i_idx] = vals 

    # Produce friendship matrix and return groups
    index = np.where(np.logical_and(prob_dlos>Pth, dperp<=perpll))
    friendship[index]=1
    assert np.all(np.abs(friendship-friendship.T) < 1e-8), "Friendship matrix must be symmetric."

    if printConf:
        print('PFoF complete!')
    #return collapse_friendship_matrix(friendship)
    return 1+connected_components(csr_array(friendship))[1]


def numeric_integration_pfof_vectorized(zmesh, z1, sig1, z2, sig2, VL_lower, VL_upper):
    sqrt_2pi = 2.506628274631
    sqrt2 = 1.41421356237
    dz = zmesh[1] - zmesh[0]
    inv_sig1 = 1.0 / sig1
    inv_den2 = 1.0 / (sqrt2 * sig2)
    # (z1-zmesh)/sig1
    tmp1 = z1[:, None] - zmesh
    tmp1 *= inv_sig1[:, None]
    # g1pdf = exp(-0.5 * tmp1^2) / (sig1 * sqrt(2π))
    np.square(tmp1, out=tmp1)
    tmp1 *= -0.5
    np.exp(tmp1, out=tmp1)
    tmp1 *= inv_sig1[:, None] / sqrt_2pi
    g1pdf = tmp1  # rename for clarity
    z2mz = z2[:, None] - zmesh
    # argument buffers
    arg_hi = (z2mz + VL_lower[:, None]) * inv_den2[:, None]
    arg_lo = (z2mz - VL_upper[:, None]) * inv_den2[:, None]
    erf_term = sc.erf(arg_hi)
    erf_term -= sc.erf(arg_lo)
    # --- final integration ---
    # P12 = 0.5 * dz * sum(g1pdf * erf_term, axis=1)
    P12 = 0.5 * dz * np.einsum("ij,ij->i", g1pdf, erf_term)
    return P12


