import numpy as np

def spike_count_correlation(spikes, chunk_duration, fps):
    """
    Computes the spike count correlation in the manner described in (Dahmen et al., 2019).
    This also allows us to compute lambda, the largest eigenvalue of the 
    connectivity matrix, which allows us to determine whether or not the system
    is at the critical state of the second type described in the paper.
    
    Deconvolved calcium traces can be used in place of absolute spikes,
    however in that case the spike count correlation will be scaled by
    an unknown constant.
    
    This procedure necessitates dividing the dataset into discrete chunks.
    The user can decide how long (in seconds) each chunk should be. Another
    way is to use the autocorrelation time of the system.

    Parameters
    ----------
    spikes : 2d array of floats/ints [neuron, timestep]
        Spiking activity of each neuron at each timestep. Can be absolute spikes
        or indirect measures such as deconvolved calcium traces. 
    chunk_duration : float
        The duration (in seconds) for each chunk.
    fps : float
        Sampling rate of the dataset.

    Returns
    -------
    cij : 1d array of floats
        (Pairwise) spike count correlation between neurons. 
    lamb : float
        The largest eigenvalue of the connectivity matrix. Takes the value of 1
        when the system is at the critical state of the second type.

    """
    chunk_length = int(np.ceil(chunk_duration * fps)) #in timesteps
        
    division = int(spikes.shape[1] / chunk_length)
    
    if spikes.shape[1] % division != 0:
        spikes = spikes[:,:-(spikes.shape[1]%division)] 
        #remove the last few timesteps so we get an even split,
        #i.e. each chunk has equal length
    assert spikes.shape[1] % division == 0
    
    nneurons,timesteps = spikes.shape
    spike_count_mat = np.empty((0,nneurons)) #columns have to be neurons for subsequent COV calculation
    
    #spike_count_mat compute total spike counts for each chunk for each neuron
    for i in range(division): 
        time_index = np.arange(i*chunk_length, (i+1)*chunk_length)
        mat_chunk = spikes[:,time_index]
        tot_spikes = np.sum(mat_chunk,axis=1)
        spike_count_mat = np.concatenate((spike_count_mat,tot_spikes[np.newaxis,:]))
    
    means = np.mean(spike_count_mat,axis=0) #each neuron's average across all time chunks
    broadcasted_means = np.repeat(means[np.newaxis,:],division,axis=0) 
    diff = spike_count_mat - broadcasted_means 
    #total in current chunk minus the average over all chunks for each neuron
    
    cij = []
    for i in range(nneurons):
        for j in range(i):
            cij += [np.sum(diff[:,i]*diff[:,j])]
    cij = np.array(cij)/(division-1)
    
    # compute lambda
    cbar = np.mean(cij)
    biased_delc_sq = np.mean((cij-cbar)**2)    
    a = []
    for i in range(nneurons):
        a += [np.sum(diff[:,i]**2)]
    a = np.array(a)/(division-1)
    abar = np.mean(a)
    bias = (abar**2 - cbar**2) / (division-1)
    delc_sq = biased_delc_sq - bias
    delta = np.sqrt(delc_sq)/abar
    lamb = np.sqrt(1-np.sqrt(1/(1+nneurons*delta**2)))
    
    return cij, lamb

