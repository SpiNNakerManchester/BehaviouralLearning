import numpy
import itertools
import random
import math

def convert_spike_list_to_timed_spikes(spike_list, min_idx, max_idx, tmin, tmax, tstep):
    times = numpy.array(range(tmin, tmax, tstep))
    spike_ids = sorted(spike_list)
    possible_neurons = range(min_idx, max_idx)
    spikeArray = dict([(neuron, times) for neuron in spike_ids if neuron in possible_neurons])
    return spikeArray 
       
def convert_file_to_spikes(input_file_name, min_idx=None, max_idx=None, tmin=None, tmax=None, compatible_input=True):

    data = numpy.array(numpy.loadtxt(fname=input_file_name), dtype=int) # get the array from the original text file
    if compatible_input: data = numpy.roll(data, 1, axis=1)             # swap neuron ID and time if necessary
    if min_idx is None: min_idx = numpy.fmin.reduce(data[:,0], 0)
    if max_idx is None: max_idx = numpy.fmax.reduce(data[:,0], 0) + 1
    if tmin is None: tmin = numpy.fmin.reduce(data[:,1], 0)
    if tmax is None: tmax = numpy.fmax.reduce(data[:,1], 0)
    data = data[(data[:,1]>=tmin) & (data[:,1]<tmax) & (data[:,0]>=min_idx) & (data[:,0]<max_idx),:] # filter by mins and maxes
    if data.shape == (0,): return {} # nothing left: return an empty dict.
    sort_keys = numpy.lexsort((data[:,1], data[:,0])) # otherwise sort, grouping by neuron ID then time.
    data = data[sort_keys,:]
    spiking_neurons = itertools.groupby(data, lambda x: x[0]) # and taking one group at a time#,
    spikeArray = dict([(neuron[0], numpy.array([spike_time[1] for spike_time in neuron[1]])) for neuron in spiking_neurons]) # create a dictionary indexed by neuron number of the spike times.
    return spikeArray

def loop_array(input_array, runtime=0, num_repeats=1, sampletime=0):
    spikeArray = {}
    for neuron in input_array:
        if not sampletime:
           sampletime = int(numpy.fmax.reduce(input_array[neuron],0))
        last_array = numpy.array([])
        if sampletime*num_repeats < runtime or (runtime > 0 and sampletime*num_repeats > runtime):
           num_repeats = runtime/sampletime
           last_array = input_array[neuron][input_array[neuron] <= (runtime%sampletime)]
        spikeArray[neuron] = numpy.concatenate([input_array[neuron]+repeat*sampletime for repeat in range(num_repeats)])
        if len(last_array): spikeArray[neuron] = numpy.concatenate([spikeArray[neuron], last_array])
    return spikeArray 

def splice_arrays(input_arrays, input_times=None, input_neurons=None):
    spikeArray = {}
    if input_neurons is None: input_neurons = [None]*len(input_arrays)
    if input_times is None: input_times = [[(reduce(lambda x, y: min(x, numpy.fmin.reduce(y,0)), input_group.values(), 0), reduce(lambda x, y: max(x, numpy.fmax.reduce(y,0)), input_group.values(), 0))] for input_group in input_arrays]
    for in_idx in range(len(input_arrays)):
        for neuron in input_arrays[in_idx].items():
            if input_neurons[in_idx] is None or neuron[0] in input_neurons[in_idx]:
               for time_range in input_times[in_idx]:
                   if time_range is None: time_range = (reduce(lambda x, y: min(x, numpy.fmin.reduce(y,0)), input_arrays[in_idx].values(), 0), reduce(lambda x, y: max(x, numpy.fmax.reduce(y,0)), input_arrays[in_idx].values(), 0))
                   if neuron[0] in spikeArray:
                      spikeArray[neuron[0]].extend([time for time in neuron[1] if time >= time_range[0] and time < time_range[1]])
                   else:
                      spikeArray[neuron[0]] = [time for time in neuron[1] if time >= time_range[0] and time < time_range[1]] 
    for neuron in spikeArray.items():
        spikeArray[neuron[0]] = numpy.sort(numpy.unique(numpy.array(neuron[1])))
    return spikeArray     
            
def splice_files(input_files, input_times=None, input_neurons=None, compatible_input=True):
    # splice_files expects a list of files, a list of lists, one for each file, giving the onset
    # and offset times for each file, and a list of neurons relevant to each file, which will be
    # spliced together into a single spike list.
    spikeArray = {}
    if input_times is None: input_times = [[(None, None)] for file_idx in len(input_files)]
    for file_idx in len(input_files):
        if input_neurons is None or input_neurons[file_idx] is None:
           max_neuron_id = numpy.fmax.reduce(input_files[file_idx].keys(), 0) + 1
           min_neuron_id = numpy.fmin.reduce(input_files[file_idx].keys(), 0)
        else:
           max_neuron_id = numpy.fmax.reduce(input_neurons[file_idx], 0) + 1
           min_neuron_id = numpy.fmin.reduce(input_neurons[file_idx], 0)
        for time_range in input_times[file_idx]:
            for neuron in convert_file_to_spikes(input_file_name=input_files[file_idx], min_idx=min_neuron_id, max_idx=max_neuron_id, tmin=time_range[0], tmax=time_range[1], compatible_input=compatible_input).items():
                if neuron[0] in spikeArray: 
                   spikeArray[neuron[0]].append(neuron[1])
                else: 
                   spikeArray[neuron[0]] = neuron[1]
    for neuron in spikeArray.items():
        spikeArray[neuron[0]] = numpy.sort(numpy.unique(numpy.array(neuron[1])))
    return spikeArray 
          
def subsample_spikes_by_time(spikeArray, start, stop, step):
    subsampledArray = {}
    for neuron in spikeArray:
        times = numpy.sort(spikeArray[neuron][(spikeArray[neuron] >= start) & (spikeArray[neuron] < stop)])
        interval = step/2 + step%2
        t_now = times[0] 
        t_start = times[0]
        t_last = len(times)
        t_index = 0
        subsampled_times = []
        while t_index < t_last:
              spikes_in_interval = 0
              while t_index < t_last and times[t_index] <= t_start + interval:
                    spikes_in_interval += 1
                    if spikes_in_interval >= interval:
                       t_start = times[t_index] + interval
                       subsampled_times.append(times[t_index])
                       try:
                          t_index = next(i for i in range(t_index, t_last) if times[i] >= t_start)
                       except StopIteration:
                          t_index = t_last
                       break
                    t_index += 1
              else:
                    if t_index < t_last:
                       t_start = times[t_index]
        subsampledArray[neuron] = numpy.array(subsampled_times)
    return subsampledArray

def random_skew_times(spikeArray, skewtime, seed=3425670):
    random.seed(seed)
    #return dict([(neuron, [int(abs(t+random.uniform(-skewtime, skewtime))) for t in spikeArray[neuron]]) for neuron in spikeArray])
    spikeDict = dict([(neuron, numpy.array(numpy.fabs(spikeArray[neuron]+numpy.random.uniform(-skewtime, skewtime, len(spikeArray[neuron]))), dtype=int)) for neuron in spikeArray])
    #test_out = open('spikeArray.txt', 'w')
    #test_out.write('%s' % spikeDict)
    #test_out.close()
    return spikeDict

def generate_shadow_spikes(spikeArray, dim_x, dim_y, move_velocity):
    """
      generate a second set of spikes as if coming from a DVS retina.
      imagines that the offset pixels perfectly register with perfect 
      timing precision.
      args: spikeArray, in the format above for an array of Spikes: a 
            dict of {id: [times]}
            dim_x, size of the field in the x-dimension
            dim_y, size of the field in the y-dimension
            move_velocity: an (s, theta) tuple, where s is the speed
            measured in pixels/ms, theta the angle of virtual movement
            measured in radians anticlockwise (0 is horizontal movement
            to the right). The function will displace the shadow by 
            s pixels in the reverse of direction indicated for each time
            point where spikes are registered. It will add one last set
            of spikes at time tmax+1, at position of the source spikes at
            time tmax.
    """
    motion_x = -move_velocity[0]*math.cos(move_velocity[1])
    motion_y = -move_velocity[0]*math.sin(move_velocity[1])
    spikeArray_out = dict([(int(motion_x+spike[0]%dim_x)+dim_x*int(motion_y+spike[0]/dim_x), spike[1][1:]) for spike in spikeArray.items() if len(spike[1]) > 1])
    spikeArray_out = dict([item for item in spikeArray_out.items() if item[0] >= 0 and item[0] < dim_x*dim_y])
    spikeArray_out.update(dict([(spike[0], numpy.append(spikeArray_out.get(spike[0], numpy.array([], dtype=int)), [int(max(spike[1]))])) for spike in spikeArray.items()]))
    return spikeArray_out
     
