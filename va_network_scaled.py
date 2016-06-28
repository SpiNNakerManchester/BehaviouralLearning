#!/usr/bin/python
"""
Visual selection model
Francesco Galluppi, Kevin Brohan

--------------------------------------------------------------------------------

Modified 2013 ADR. WTA sharpened (bug-fixed?), added plasticity (optional)
between V2 and V4, added PFC preferred and aversive stimulus capability
with tunable parameters

--------------------------------------------------------------------------------

Scaling module added January 2014 ADR to auto-scale dependent network parameters
with the size of the input field.

--------------------------------------------------------------------------------

Enhanced June 2014 ADR. Modified Gaussian creation to permit specification of
gains and eccentricities in the Gaussian filter (also available for Gabor filtering).
Tuned weights, gain, eccentricity for sample visual input.
Added optional features that:

1) Allow the PFC to provide bipolar (excitatory/inhibitory) reinforcement
2) Implement top-down feedback as per Bernabe Linares-Barranco's suggestions
3) Add additional PFC priming to V1 layer
4) Provide a new output module for better visualisation of LIP output

--------------------------------------------------------------------------------

FEF style PFC added September 2014 ADR. The "active_pfc" option instantiates a
module which makes PFC output dependent upon V2 activity, modelling the FEF
in biology.

--------------------------------------------------------------------------------

Multistage PFC added October 2015 ADR. "multistage_pfc" adds a fixed-bias input 
the FEF-style PFC which sets the initial preference. It can also be used to turn
off any preference altogether. Note that the fixed-input layer of the PFC needs
to be generating spikes in order for the FEF to become active!

--------------------------------------------------------------------------------

This is the scalable PyNN only version of the model and is designed to
work with real input from iCub cameras using Yarp for comms

"""
# general Python imports
import sys
import itertools
import re
import copy
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.colors import ListedColormap

# utility module imports for this script
import visual_network_scaling
import visual_metrics

from gaussiancreatejose import *
from vector_topographic_activity_plot import * #mapped_boxplot
from spike_file_to_spike_array import * # convert_file_to_spikes
from transform_mapped_inputs import move_objects # enable moving objects 

# pyNN imports
#from pyNN.brian import *               # use if running brian
from pyNN.random import *               # add support for RNG from native PyNN
from pyNN.spiNNaker import *            # Imports the pyNN.spiNNaker module 
from pyNN.utility import Timer

from operator import itemgetter

time_step = 1.0
# Simulation Setup
setup(timestep=time_step, min_delay = 1.0, max_delay = 11.0, db_name='vis_attn_iCub.sqlite')

# Fixes for overconcentration of spikes with synchronous firing
set_number_of_neurons_per_core(IF_curr_exp, 228)
set_number_of_neurons_per_core(IF_curr_dual_exp, 64)

# Workaround to fix ReverseIPTagMulticastSource limitations on number of 
# spikes/ms it can handle.
set_number_of_neurons_per_core(SpikeSourceArray, 256)           

layer_to_observe = 'lip'
record_lip_stats = False
lip_map_on = True
plasticity_on = True
feedback_on = True
aversive_inhibitory = True
top_down_priming = False
metrics_on = False
vector_plot = True
active_pfc = True      # PFC as FEF
multistage_pfc = False # switchable PFC. Should have more biologically relevant name.
output_to_file = False
randomise_spike_times = True
compatible_output = True # added parameter for getSpikes order. True = (id, time)

preferred_orientation = 0   # ADR added preferred and aversive orientations. These 
aversive_orientation = 2   # will be used to set biassing in LIP
base_num_neurons = 32 #64 #128
weight_prescale_factor = 1  # prescales weights and capacitances to account for system limits
subsample_step = 6 #1 # number of milliseconds to take average spike counts for spike subsampling
random_spike_jitter = 3 # +- timing on any given spike 

input_file_type = 2 #1 # Input from: 0 - a simple list of ids; 1 - a single full list of spikes and times; 2 - a pair of polarities with spikes and times
metrics_with_input = 0 # Get metrics from 1 - the input file; 0 - a separate file

input_file_base_names = ['./run3'] # base name of the input file(s)
stimulus_onset_offsets = [[(0, 100)]]  
# some examples of how to use these parameters with more complex input
#input_file_base_names = ['./TestData/horiz_left', './TestData/vert_right'] # base name of the input file(s)
#stimulus_onset_offsets = [[(0, 500), (1000, 2000), (2500, 3000), (4000, 5000)], [(500, 1000), (2000, 2500), (3500, 4500)]]

compatible_input = True # whether input has (time, spike) order or reverse. True = (time, spike)
moving_objects = False # whether objects will have motions
random_motions = True # and whether those motions are random
motion_period = 100 # mean spacing between move events in random motion
motion_time_jitter = 25 # +- interval for random motion components
virtual_velocity = (3, math.pi/2) # virtual velocity (displacement, angle) to use when computing shadow distance
save_weights = False

if input_file_type == 2:
   InputFilesPol1 = ['%s_%dx%d_pol1.dat' % (base_name, base_num_neurons, base_num_neurons) for base_name in input_file_base_names]
   InputFilesPol2 = ['%s_%dx%d_pol2.dat' % (base_name, base_num_neurons, base_num_neurons) for base_name in input_file_base_names]
else:
   InputFilesPol1 = ['%s_%dx%d' % (base_name, base_num_neurons, base_num_neurons) for base_name in input_file_base_names]

if metrics_on:
   if metrics_with_input:
      MetricFile = InputFilePol1
   else:
      MetricFile = 'object_annots_%dx%d' % (tuple([base_num_neurons]*2))
	
scale = visual_network_scaling.scale_factor(dimensions=2, dim_sizes=None, base_dim_size=base_num_neurons, subsample_factor_i_1=1.6, subsample_factor_1_2=1.0, subsample_factor_2_4=2.0, subsample_factor_4_L=1.0, subsample_factor_4_P=1.0, base_filter_size=5.0, sweep_filter=False, filter_scale=(1,1,1), xy_kernel_quant_cutoff=(0,0), base_v2_subfield_size=2, sweep_subfield=False, subfield_scale=(1,1,1), active_P=active_pfc)
scale.setindex(f_idx=0, s_idx=0)

# Create some arrays for neuron ID to x-y coord mappings
# for doing nice 2D plotting

input_x_array = numpy.zeros((int(scale.input_size[0]*scale.input_size[1])),int)
input_y_array = numpy.zeros((int(scale.input_size[0]*scale.input_size[1])),int)

startx = 0
starty = 0

for nrn in xrange(scale.input_size[0]*scale.input_size[1]):
   input_x_array[nrn] = startx
   input_y_array[nrn] = starty
   if startx == (scale.input_size[0] - 1):
       startx = 0
       starty = starty + 1
   else:
       startx = startx + 1
      
lip_x_array = numpy.zeros((int(scale.lip_pop_size[0]*scale.lip_pop_size[1])),int)
lip_y_array = numpy.zeros((int(scale.lip_pop_size[0]*scale.lip_pop_size[1])),int)

startx = 0
starty = 0

for nrn in xrange(scale.lip_pop_size[0]*scale.lip_pop_size[1]):
   lip_x_array[nrn] = startx
   lip_y_array[nrn] = starty
   if startx == (scale.lip_pop_size[0] - 1):
       startx = 0
       starty = starty + 1
   else:
       startx = startx + 1

#PARAMETERS gaussian
scales=2                # scales
orientations=4          # orientations
# use these if not using the visual_network_scaling module 
#sizeg1=5.0             # size of the gaussian filter

#size_k1 = 5            # x-Size kernel (connections)
#size_k2 = 5            # y-Size kernel (connections)
#jump = 1               # overlapping 
delays = 1.0            # connection delays
v2_v4_delays = 2*delays if active_pfc else delays

# PARAMETERS NETWORK

pfc_bias_pref = 0.825 if active_pfc else 1.0 # strong stimulation; should cause persistent spiking ADR
pfc_bias_avert = 0.7 if active_pfc else 0.6  # mild stimulation to non-aversive orientations; 
                                             # biasses away from the  aversive group. ADR
pfc_bias_neut = 0.5 if active_pfc else 0.5   # very mild stimulation; 
                                             # should just keep a population in random spiking ADR
feedback_weight_scaling = 0.81 
input_gain = 1.0                             # steepness of the Gaussian input filter. 
                                             # This varies parametrically with scales. 
                                             # Generally the more scales, the shallower. 
gaussian_eccentricity = 5.5                  # ratio of major/minor axis for the input filter. 
                                             # This varies parametrically with orientations. 
                                             # Generally the more orientations, the more eccentric.
input_strength = 2.25*weight_prescale_factor # drive strength of the input->gaussian filters
v1_v2_weights = 8*weight_prescale_factor     # weights between v1 and v2 orientation maps (one2one connectors)
wta_v2 = True      	                     # do the gaussian filter maps in v2 inhibit each other?
wta_between_v2_weight = -1.85*weight_prescale_factor # inhibition weight between several orientations in v2 
wta_within_v2_weight = -0.6*weight_prescale_factor   # inhibition weight within a single orientation in v2 
weights_v4_lip = 22*weight_prescale_factor 
wta_lip_weight = -7*weight_prescale_factor           # competition in LIP
pfc_pre_weights = 0.2*weight_prescale_factor   # pfc_pre->pfc gating weights. These multiply the bias.
pfc_v1_weights = 0.0725*weight_prescale_factor # pfc->v1 competition biasing weights. 
                                               # Used if top-down priming is on.
pfc_v4_weights = 7.25*weight_prescale_factor if active_pfc else 0.0725*weight_prescale_factor # pfc->v4 competition biasing weights
if multistage_pfc:
   v2_pfc_weights = 20*weight_prescale_factor
else:
   v2_pfc_weights = 48*weight_prescale_factor # base weight value for v1-pfc connections when using active PFC
wta_bias = 1.3 # sets the relative strength of the other- (heterosynaptic) vs self- (homosynaptic) inhibitory connections for WTA
wta_IOR_delay = 1           # scales the delay for IOR (self-inhibition)

# set weight values for the critical V2->V4 connection. 
if plasticity_on:
   weights_v2_v4 = 5*weight_prescale_factor 
elif active_pfc:
   if feedback_on:
      weights_v2_v4 = 3*weight_prescale_factor 
   else:
      weights_v2_v4 = 3*weight_prescale_factor 
else:
   if feedback_on:
      weights_v2_v4 = 4.5*weight_prescale_factor 
   else:
      weights_v2_v4 = 5.5*weight_prescale_factor  

# random objects and initialisations
rng = NumpyRNG(seed=28374)
v_init_distr = RandomDistribution('uniform', [-55,-95], rng)
v_rest_distr = RandomDistribution('uniform', [-55,-65], rng)
if moving_objects and random_motions: # allow for random object motions
   t_move_distrs = []
   dx_distrs = []
   dy_distrs = []
   for input_file_idx in range(len(input_file_base_names)): 
       t_move_distrs.append(RandomDistribution('uniform', [-motion_time_jitter, motion_time_jitter], rng))
       dx_distrs.append(RandomDistribution('uniform', [-1, 1], rng))
       dy_distrs.append(RandomDistribution('uniform', [-1, 1], rng))
      
# time parameters
runtime = 100   # set for short run 
#runtime = 5000 # set for a longer run
plot_time_window = False # Enable plot time windowing to reduce graph size/detail
p_t_w = (3000, 3200) # (min, max) times for plotting
plot_t_max = p_t_w[1] if plot_time_window else runtime # plot max time
plot_t_min = p_t_w[0] if plot_time_window else 0 # plot min time
metric_window = 100
metric_start_offset = 0
metric_t_start = 0
metric_t_stop = runtime

# moving object parameters
if moving_objects:
   if random_motions:
      object_motions = []
      random_exclude_list = [2]
      for input_file_idx in range(len(input_file_base_names)):
          if input_file_idx not in random_exclude_list:
             object_motions.append(dict([(int(t+next(t_move_distrs[input_file_idx])), (int(base_num_neurons*next(dx_distrs[input_file_idx])), int(base_num_neurons*next(dy_distrs[input_file_idx])))) for t in range(0, 500, motion_period)]))
          else:
             object_motions.append({})
   else:         
      object_motions = [{100: (int(base_num_neurons*0.3), int(base_num_neurons*0.4)), 200: (int(base_num_neurons*-0.5), int(base_num_neurons*0.4)), 300: (int(base_num_neurons*-0.5), int(base_num_neurons*-0.2)), 400: (int(base_num_neurons*0.3), int(base_num_neurons*-0.2))}, {50: (int(base_num_neurons*0.7), int(base_num_neurons*0.1)), 150: (int(base_num_neurons*-0.3), int(base_num_neurons*0.1)), 250: (int(base_num_neurons*-0.3), int(base_num_neurons*-0.8)), 350: (int(base_num_neurons*0.7), int(base_num_neurons*-0.8)), 450: (0, 0)}, {}]


# Neural Parameters
tau_m    = 24.0    # (ms)
cm       = 1
v_rest   = -65      # (mV)
v_thresh = -45      # (mV)
v_reset  = -65      # (mV)
t_refrac = 3.       # (ms) (clamped at v_reset)
tau_syn_exc = 3
tau_syn_inh = tau_syn_exc*3
i_offset = 0

if multistage_pfc:
   tau_syn_exc_2 = 100 # long time contant creates an effective input integrator.
   i_bias_pref = 0   # with multistage areas input spike rate into stage 1
   i_bias_avert = 0  # will set the PFC bias, not i_bias
   i_bias_neut = 0
   pfc_on_time = runtime/2
   pfc_prescalar = 1 # pfc_pre spike-frequency prescalar
   pfc_pre_times = [None for i in range(orientations)]
   for i in range(orientations): # instantiate arrays for PFC activation. These could vary based upon orientation preference
       if i == preferred_orientation:
          pfc_pre_times[i] = [[t for t in itertools.ifilterfalse(lambda s: s%pfc_prescalar, range(pfc_on_time))] for n in range(scale.pfc_pop_size[0]*scale.pfc_pop_size[1])]
       elif i == aversive_orientation:
          pfc_pre_times[i] = [[t for t in itertools.ifilterfalse(lambda s: s%pfc_prescalar, range(pfc_on_time))] for n in range(scale.pfc_pop_size[0]*scale.pfc_pop_size[1])]
       else:
          pfc_pre_times[i] = [[t for t in itertools.ifilterfalse(lambda s: s%pfc_prescalar, range(pfc_on_time))] for n in range(scale.pfc_pop_size[0]*scale.pfc_pop_size[1])]
else:
   i_bias_pref = pfc_bias_pref   # with single PFC biasses are literal currents into the neuron ADR
   i_bias_avert = pfc_bias_avert #                                                                               
   i_bias_neut = pfc_bias_neut   #

if plasticity_on: # set plasticity between v2 and v4, if desired
   # sets symmetric window, biassed slightly towards inhibition,
   # maximum weight is the must-fire weight
   stdp_model = STDPMechanism(
                timing_dependence = SpikePairRule(tau_plus = 30.0, tau_minus = 30.0),
                weight_dependence = AdditiveWeightDependence(w_min = 0, w_max = 20, A_plus=0.01, A_minus = 0.012) 
                )

timer = Timer()
timer.start()


# cell_params will be passed to the constructor of the Population Object

cell_params = {
    'tau_m'      : tau_m,    'cm'         : cm,    
    'v_rest'     : v_rest,   'v_reset'    : v_reset,  'v_thresh'   : v_thresh,
    'tau_syn_E'       : tau_syn_exc,        'tau_syn_I'       : tau_syn_inh, 'tau_refrac'       : t_refrac, 'i_offset' : i_offset
    }

# population and projection containers
v1_pop = []
v2_pop = []
v4_pop = []
pfc_pre = []
pfc = []
projections = []
if save_weights:
   learning_projections = {}

print "%g - Creating input population: %d x %d" % (timer.elapsedTime(), scale.input_size[0], scale.input_size[1])
data_input_1_components = []
for input_file_num in range(len(input_file_base_names)):
    input_file = open(InputFilesPol1[input_file_num], 'r')
    if not input_file_type:
       input_spike_list = eval(input_file.readline())
       data_input_1 = convert_spike_list_to_timed_spikes(spike_list=input_spike_list, tmax=runtime, tstep=int(time_step))
       sample_time = runtime
    else:
       data_input_1 = convert_file_to_spikes(input_file_name=InputFilesPol1[input_file_num], tmax=runtime, compatible_input=compatible_input)
       try:
          sample_time = int(re.search('^#\s*runtime\s*=\s*(\d)\s*', input_file.read(), flags=re.MULTILINE).group(1))
       except AttributeError: 
          sample_time = reduce(lambda x, y: max(x, numpy.fmax.reduce(y,0)), data_input_1.values(), 0)
    input_file.close()
    data_input_1 = subsample_spikes_by_time(data_input_1, 0, runtime, subsample_step)
    if moving_objects: 
       data_input_1 = move_objects(data_input_1, base_num_neurons, base_num_neurons, object_motions[input_file_num]) 

    if sample_time < runtime:
       data_input_1 = loop_array(input_array=data_input_1, runtime=runtime, sampletime=sample_time)
    if randomise_spike_times:
       data_input_1 = random_skew_times(data_input_1, random_spike_jitter)
    data_input_1_components.append(data_input_1)
data_input_1 = splice_arrays(input_arrays=data_input_1_components, input_times=stimulus_onset_offsets)
input_1_times = [data_input_1[neuron] if neuron in data_input_1 else [] for neuron in range(scale.input_size[0]*scale.input_size[1])]

input_pol_1 = Population(scale.input_size[0]*scale.input_size[1],         # size 
              SpikeSourceArray,   # Neuron Type
                         {'spike_times': input_1_times},   # Neuron Parameters
              label="input_pol_1") # Label
if layer_to_observe == 'input_pol_1' or layer_to_observe == 'all':
   print "%g - observing input (positive polarity)" % timer.elapsedTime()    
   input_pol_1.record() # ('spikes', to_file=False)

data_input_2_components = []
for input_file_num in range(len(input_file_base_names)):
    if input_file_type < 2:
       data_input_2 = generate_shadow_spikes(data_input_1_components[input_file_num], scale.input_size[0], scale.input_size[1], virtual_velocity)
    else:
       input_file = open(InputFilesPol2[input_file_num], 'r')
       data_input_2 = convert_file_to_spikes(input_file_name=InputFilesPol2[input_file_num], tmax=runtime, compatible_input=compatible_input)
       try:
          sample_time = int(re.search('^#\s*runtime\s*=\s*(\d)\s*', input_file.read(), flags=re.MULTILINE).group(1))
       except AttributeError: 
          sample_time = reduce(lambda x, y: max(x, numpy.fmax.reduce(y,0)), data_input_2.values(), 0)
       data_input_2 = subsample_spikes_by_time(data_input_2, 0, runtime, subsample_step)
    if moving_objects: 
       data_input_2 = move_objects(data_input_2, base_num_neurons, base_num_neurons, object_motions[input_file_num])
       if sample_time < runtime:
          data_input_2 = loop_array(input_array=data_input_2, runtime=runtime, sampletime=sample_time)
    input_file.close()
    # for spikes originating in a file type less than 2 the randomisation 
    # will be carried out twice. If this causes a problem, randomisation of 
    # times for data_input_1 can be carried out outside the file number loop.
    if randomise_spike_times:
       data_input_2 = random_skew_times(data_input_2, random_spike_jitter)
    data_input_2_components.append(data_input_2)
data_input_2 = splice_arrays(input_arrays=data_input_2_components, input_times=stimulus_onset_offsets)
input_2_times = [data_input_2[neuron] if neuron in data_input_2 else [] for neuron in range(scale.input_size[0]*scale.input_size[1])]

input_pol_2 = Population(scale.input_size[0]*scale.input_size[1],         # size 
              SpikeSourceArray,   # Neuron Type
                         {'spike_times': input_2_times},   # Neuron Parameters
              label="input_pol_2") # Label
if layer_to_observe == 'input_pol_2' or layer_to_observe == 'all':
   print "%g - observing input (negative polarity)" % timer.elapsedTime()    
   input_pol_2.record() # ('spikes', to_file=False) 

print "%g - Creating v1 populations" % timer.elapsedTime()

for i in range(orientations):           # Cycles orientations
    # creates a population for each connection
    v1_pop.append(Population(scale.v1_pop_size[0]*scale.v1_pop_size[1],         # size 
                  IF_curr_exp,   # Neuron Type
                  cell_params,   # Neuron Parameters
                  label="v1_%d" % i)) # Label)    
    if layer_to_observe == 'v1' or layer_to_observe == 'all':
        print "%g - observing v1" % timer.elapsedTime()
        v1_pop[i].record() # ('spikes', to_file=False)       

print "%g - Creating v2 populations" % timer.elapsedTime()

for i in range(orientations):           # Cycles orientations
    # creates a population for each connection
    v2_pop.append(Population(scale.v2_pop_size[0]*scale.v2_pop_size[1],         # size 
                  IF_curr_exp,   # Neuron Type
                  cell_params,   # Neuron Parameters
                  label="v2_%d" % i)) # Label)      
    if layer_to_observe == 'v2' or layer_to_observe == 'all':
        print "%g - observing v2" % timer.elapsedTime()    
        v2_pop[i].record() # ('spikes', to_file=False)   

print "%g - Creating v4 populations" % timer.elapsedTime()

for i in range(orientations):           # Cycles orientations
    # creates a population for each connection
    v4_pop.append(Population(scale.v4_pop_size[0]*scale.v4_pop_size[1],         # size 
	          IF_curr_exp,   # Neuron Type
		  cell_params,   # Neuron Parameters
		  label="v4_%d" % i)) # Label)      
    if layer_to_observe == 'v4' or layer_to_observe == 'all':                     
        print "%g - observing v4" % timer.elapsedTime()    
        v4_pop[i].record() # ('spikes', to_file=False)     

print "%g - Creating PFC populations" % timer.elapsedTime()

for i in range(orientations):           # Cycles orientations
    if multistage_pfc:
       pfc_pre.append(Population(scale.pfc_pop_size[0]*scale.pfc_pop_size[1], # size 
                      SpikeSourceArray,               # Neuron Type
                      {'spike_times': pfc_pre_times[i]}, # Neuron Parameters
                      label="pfc_pre_%d" % i))        # Label
       if layer_to_observe == 'pfc_pre' or layer_to_observe == 'all':                     
          print "%g - observing pfc_pre" % timer.elapsedTime()    
          pfc_pre[i].record()
       cell_params_dual_exp = copy.deepcopy(cell_params)
       cell_params_dual_exp['tau_syn_E2'] = tau_syn_exc_2
       pfc.append(Population(scale.pfc_pop_size[0]*scale.pfc_pop_size[1],        # size 
	          IF_curr_dual_exp,   # Neuron Type
	          cell_params,   # Neuron Parameters
	          label="pfc_%d" % i))  
    else:
       pfc.append(Population(scale.pfc_pop_size[0]*scale.pfc_pop_size[1],        # size 
	          IF_curr_exp,   # Neuron Type
	          cell_params,   # Neuron Parameters
	          label="pfc_%d" % i))
       
    pfc[i].randomInit(v_init_distr)
    # set biasses to hardwire preference ADR
    if i == preferred_orientation:
       pfc[i].set('i_offset', i_bias_pref)
    elif i == aversive_orientation:
       pfc[i].set('i_offset', i_bias_avert)
    else:
       pfc[i].set('i_offset', i_bias_neut)
    if active_pfc == False:
       v_rest_or = []
       for j in range(pfc[i].size):
	   v_rest_or.append(v_rest_distr.next())
       pfc[i].set('v_rest', v_rest_or)
	    
    if layer_to_observe == 'pfc' or layer_to_observe == 'all':                     
        print "%g - observing pfc" % timer.elapsedTime()    
        pfc[i].record() # ('spikes', to_file=False)  


print "%g - Creating LIP population" % timer.elapsedTime()
lip = Population(scale.lip_pop_size[0]*scale.lip_pop_size[1],         # size 
  	         IF_curr_exp,   # Neuron Type
	         cell_params,   # Neuron Parameters
	         label="lip")
if layer_to_observe == 'lip' or layer_to_observe == 'all':                     
    print "%g - observing lip" % timer.elapsedTime()
    lip.record() # ('spikes', to_file=False) 


print "%g - Creating gaussian Filters connections: scale=%d orientation=%d size=%f" % (timer.elapsedTime(), scales, orientations, scale.filter_scale)
gaussian_filters = TunedGaussianConnectorList(scales, orientations, scale.filter_scale, input_gain, gaussian_eccentricity)

for i in range(orientations):
    # creates connections lists for different orientations, implementing a 
    # convolutional network with different gaussian orientation filters (single scale)
    conn_list = Filter2DConnector_jose(scale.input_size[0], scale.input_size[1], 
		                       scale.v1_pop_size[0], scale.v1_pop_size[1], 
		                       gaussian_filters[i], 
		                       scale.x_kernel, scale.y_kernel, 
		                       scale.jump[0], delays, 
		                       gain=input_strength)
    projections.append(Projection(input_pol_1, v1_pop[i], 
	FromListConnector(conn_list), label='input[p0]->v1_pop_%d' % (i)))
    projections.append(Projection(input_pol_2, v1_pop[i], 
	FromListConnector(conn_list), label='input[p1]->v1_pop_%d' % (i)))

if multistage_pfc == True:
   print "%g - Creating pfc_pre->pfc connections" % timer.elapsedTime()
   for i in range(orientations):
             if i == preferred_orientation: 
                pfc_in_weight = pfc_pre_weights*pfc_bias_pref
             elif i == aversive_orientation:
                pfc_in_weight = pfc_pre_weights*pfc_bias_avert
             else:
                pfc_in_weight = pfc_pre_weights*pfc_bias_neut
             projections.append(Projection(pfc_pre[i], 
                                           pfc[i], 
                                           OneToOneConnector(weights=pfc_in_weight, delays=delays), 
                                           target='excitatory2'))

if active_pfc == True:
   print "%g - Creating v2->pfc connections" % timer.elapsedTime()
   pfc_filters = TunedGaussianConnectorList(1, 1, scale.pfc_filter_scale[0], scale.pfc_filter_gain, scale.pfc_eccentricities[0])
   pfc_filter_conn_list = Filter2DConnector_jose(scale.v2_pop_size[0], scale.v2_pop_size[1], 
		                       scale.pfc_pop_size[0], scale.pfc_pop_size[1], 
		                       pfc_filters[0], 
		                       int(math.floor(scale.pfc_filter_scale[0])), int(math.floor(scale.pfc_filter_scale[1])), 
		                       scale.pfc_jumps[0], delays, 
		                       gain=v2_pfc_weights)
   for i in range(orientations):
       projections.append(Projection(v2_pop[i], pfc[i], FromListConnector(pfc_filter_conn_list), label='v2_pop%d->pfc_%d' % (i,i)))
   
print "%g - Creating v1->v2 connections" % timer.elapsedTime()

for i in range(orientations):
    projections.append(Projection(v1_pop[i], v2_pop[i], OneToOneConnector(weights=v1_v2_weights, delays=delays), label='v1->v2(pop_%d)' % (i)))
    if feedback_on:
       projections.append(Projection(v2_pop[i], v1_pop[i], OneToOneConnector(weights=v1_v2_weights*feedback_weight_scaling, delays=delays), label='v2->v1(pop%d)' % (i)))

if(wta_v2 == True):
    print "%g - Creating Lateral inhibition for the v2 populations" % timer.elapsedTime()
    for i in range(orientations):           # Cycles orientations
	for j in range(orientations):           # Cycles orientations
	    if (i!=j):                      # Avoid self connections
		# Creates lateral inhibition between the v2 populations
		print "%g - v2[%d]->v2[%d] lateral inhibition" % (timer.elapsedTime(), i, j)
          	wta_between_list =  ProximityConnector(scale.v2_pop_size[0], scale.v2_pop_size[1], scale.v2_subfield, 
                                                        wta_between_v2_weight, 1, allow_self_connections=True)
		projections.append(Projection(  v2_pop[i], 
		                                v2_pop[j], 
		                                FromListConnector(wta_between_list), 
		                                target='inhibitory'))                    

print "%g - Creating within inhibition pools" % timer.elapsedTime()
for i in range(orientations):           # Cycles orientations
    wta_within_list =  ProximityConnector(scale.v2_pop_size[0], scale.v2_pop_size[1], scale.v2_subfield, 
                                            wta_within_v2_weight, 1, allow_self_connections=False)
    print "%g - v2[%d] within inhibition" % (timer.elapsedTime(), i)
    projections.append(Projection(  v2_pop[i], 
		                    v2_pop[i], 
		                    FromListConnector(wta_within_list), 
		                    target='inhibitory'))                    
				
print "%g - Creating v2->v4 projections" % timer.elapsedTime()

for i in range(orientations):           # Cycles orientations
    v2_v4_conn_list =  subSamplerConnector2D(scale.v2_pop_size[0], scale.v4_pop_size[0], weights_v2_v4, v2_v4_delays)
    print "%g - v2-v4[%d] subsampling projection" % (timer.elapsedTime(), i)
    if plasticity_on: # added ability to set plasticity ADR
       # this turns on plasticity in the last projection to be appended
       # to the list (which was just done above)
       Proj_Plasticity = SynapseDynamics(slow=stdp_model)
       projections.append(Projection(  v2_pop[i], 
		                       v4_pop[i], 
		                       FromListConnector(v2_v4_conn_list),
                                       synapse_dynamics = Proj_Plasticity,
		                       target='excitatory'))
    else:
       projections.append(Projection(  v2_pop[i], 
		                       v4_pop[i], 
		                       FromListConnector(v2_v4_conn_list),
		                       target='excitatory'))
    if save_weights:
       learning_projections[len(projections)-1] = "v2-v4[%d]" % i
    if feedback_on: # feedback adds top-down biassing of preferred/active stimuli
       v4_v2_conn_list =  overSamplerConnector2D(scale.v4_pop_size[0], scale.v2_pop_size[0], weights_v2_v4*feedback_weight_scaling, v2_v4_delays) # overSamplerConnector remaps the downscaled connections to their original sources
       projections.append(Projection(  v4_pop[i], 
                                       v2_pop[i], 
                                       FromListConnector(v4_v2_conn_list),
                                       target='excitatory'))

print "%g - Creating v4->lip projections" % timer.elapsedTime()
for i in range(orientations):           # Cycles orientations
    projections.append(Projection(  v4_pop[i], 
		                    lip, 
		                    OneToOneConnector(weights=weights_v4_lip, delays=delays), 
		                    target='excitatory'))        

print "%g - Creating LIP WTA" % timer.elapsedTime()

# ADR added WTA connections to neighbouring neurons. Original version had only the
# self connection, which looks wrong (would be an "inverse WTA")
#projections.append(Projection(  lip, 
#		                lip, 
#		                AllToAllConnector(weights=wta_lip_weight*wta_bias, delays=delays, allow_self_connections=False),  
#		                target='inhibitory'))

# Temporary workaround for PACMAN103 builds a FromListConnector until such time as allow_self_connections
# option is properly supported
lip_WTA_conn_list = [(i, j, wta_lip_weight if i == j else wta_lip_weight*wta_bias, 1) for i in range(scale.lip_pop_size[0]*scale.lip_pop_size[1]) for j in range(scale.lip_pop_size[0]*scale.lip_pop_size[1])]
projections.append(Projection(  lip, 
		                lip, 
		                FromListConnector(lip_WTA_conn_list),  
		                target='inhibitory'))                       


print "%g - Creating pfc->v4 projections" % timer.elapsedTime()
for i in range(orientations):           # Cycles orientations
    if i == aversive_orientation:       # aversive orientation connectivity projects to other orientations ADR
       if aversive_inhibitory:
          if active_pfc:
             projections.append(Projection(  pfc[i], 
                                             v4_pop[i], 
                                             OneToOneConnector(weights=-pfc_v4_weights, delays=delays), 
                                             target='inhibitory'))
          else:
             projections.append(Projection(  pfc[i], 
                                             v4_pop[i], 
                                             AllToAllConnector(weights=-pfc_v4_weights, delays=delays), 
                                             target='inhibitory'))
          if top_down_priming:
             projections.append(Projection(  pfc[i],
                                             v1_pop[i],
                                             AllToAllConnector(weights=-pfc_v1_weights, delays=delays),
                                             target='inhibitory'))
       else: 
          for j in [orientation for orientation in range(orientations) if orientation != i]:
              if active_pfc:
                 projections.append(Projection(  pfc[i], 
                                                 v4_pop[j], 
                                                 OneToOneConnector(weights=pfc_v4_weights, delays=delays), 
                                                 target='excitatory'))
              else:
                 projections.append(Projection(  pfc[i], 
                                                 v4_pop[j], 
                                                 AllToAllConnector(weights=pfc_v4_weights, delays=delays), 
                                                 target='excitatory'))
              if top_down_priming:
                 projections.append(Projection(  pfc[i],
                                                 v1_pop[j],
                                                 AllToAllConnector(weights=pfc_v1_weights, delays=delays),
                                                 target='excitatory'))
    else:
       if active_pfc:
          projections.append(Projection(  pfc[i], 
                                          v4_pop[i], 
                                          OneToOneConnector(weights=pfc_v4_weights, delays=delays), 
                                          target='excitatory'))
       else:       
          projections.append(Projection(  pfc[i], 
                                          v4_pop[i], 
                                          AllToAllConnector(weights=pfc_v4_weights, delays=delays), 
                                          target='excitatory'))
       if top_down_priming:
          projections.append(Projection(  pfc[i],
                                          v1_pop[i],
                                          AllToAllConnector(weights=pfc_v1_weights, delays=delays),
                                          target='excitatory'))                    
                

lip.set('tau_syn_E', 20)

setup_time = timer.elapsedTime()

# Run the model

run(runtime)    # Simulation time

run_time = timer.elapsedTime()

# write informational/benchmarking file if desired
if output_to_file:
   output_file = open("./VA_runs_IEEE.txt", "a+")
   output_file.write("--------------------------------------------------------------------------------\n")
   output_file.write("NETWORK PARAMETERS:\n")
   output_file.write("-------------------------------\n")
   output_file.write("Input file name: %s\n" % InputFilesPol1)
   output_file.write("Network base input size: %d\n" % base_num_neurons)
   output_file.write("Feedback on? %s\n" % feedback_on)
   output_file.write("FEF PFC? %s\n" % active_pfc)
   output_file.write("Learning on? %s\n" % plasticity_on)
   output_file.write("Preferred orientation: %d\n" % preferred_orientation)
   output_file.write("Aversive orientation: %d\n" % aversive_orientation)
   output_file.write("WV2->V4_init: %f\n" % weights_v2_v4)
   output_file.write("-------------------------------\n")
   output_file.write("TIMINGS:\n")
   output_file.write("Setup time: %f s\n" % setup_time)
   print type(run_time)
   print type(setup_time)
   print type(runtime)
   output_file.write("Load time: %f s \n" % (run_time-setup_time-(runtime/1000.0)))
   output_file.write("Run time: %f s \n" % (runtime/1000.0))   
else: 
   print "Setup time", setup_time
   print "Load time", (run_time - setup_time - runtime/1000.0)
   print "Run time", (runtime/1000.0)

# output weights on learning connections if desired
if save_weights:
   weights_after = dict([(learning_projections[i], projections[i].getWeights()) for i in learning_projections])
   weights_file = open('IEEETNNLS_moving_wt_record.txt', 'w+')
   weights_file.write('weights after learning:\n')
   for i in learning_projections:
       weights_file.write('orientation %s:\n' % learning_projections[i])
       weights_file.write('%s\n' % weights_after[learning_projections[i]])
   weights_file.close()
 
# get spikes and plot

# For layers with sub-populations (V1, V2, PFC, V4)

if layer_to_observe == 'input_pol_1':
   data = numpy.asarray(input_pol_1.getSpikes())
   # data[:,0] = data[(all_rows, column_0)]
   if plot_time_window:
      plt.scatter(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]), 1], 
                  data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]), 0],
                  color='green', s=4)
   else:
      plt.scatter(data[:,1], data[:,0], color='green', s=4) # s=1
elif layer_to_observe == 'input_pol_2':
   data = numpy.asarray(input_pol_2.getSpikes())
   if plot_time_window:
      plt.scatter(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]), 1], 
                  data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]), 0],
                  color='green', s=4)
   else:
      plt.scatter(data[:,1], data[:,0], color='green', s=4) # s=1
if layer_to_observe == 'v1':
   id_accumulator=0
   data_vector = []
   for i in range(len(v1_pop)):
       data = numpy.asarray(v1_pop[i].getSpikes())
       if vector_plot:
          if plot_time_window:
             data_vector.append(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),:])
          else:
             data_vector.append(data)
       else:
          if len(data) > 0:
             if plot_time_window:
                if compatible_output:
                   plt.scatter(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),1], 
                               data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),0] 
                                   + id_accumulator, 
                               color='green', s=4) # s=1
                else:
                   plt.scatter(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),1], 
                               data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),0] 
                                   + id_accumulator, 
                               color='green', s=4) # s=1        
             else:
                if compatible_output:
                   plt.scatter(data[:,1], data[:,0] + id_accumulator, color='green', s=4) # s=1
                else:
                   plt.scatter(data[:,1], data[:,0] + id_accumulator, color='green', s=4) # s=1        
          id_accumulator = id_accumulator + v1_pop[i].size
   if vector_plot:
      mapped_arrowplot(data_vector, x_dim=scale.v1_pop_size[0], y_dim=scale.v1_pop_size[1], t_max=plot_t_max, t_min=plot_t_min, compatible_output=compatible_output)
elif layer_to_observe == 'v2':
   id_accumulator=0
   data_vector = []
   for i in range(len(v2_pop)):
       data = numpy.asarray(v2_pop[i].getSpikes())
       if vector_plot:
          if plot_time_window:
             data_vector.append(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),:])
          else:
             data_vector.append(data)
       else:
          if len(data) > 0:
             if plot_time_window:
                if compatible_output:
                   plt.scatter(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),1], 
                               data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),0] 
                                   + id_accumulator, 
                               color='red', s=4) # s=1
                else:
                   plt.scatter(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),1], 
                               data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),0] 
                                   + id_accumulator, 
                               color='red', s=4) # s=1        
             else:
                if compatible_output:
                   plt.scatter(data[:,1], data[:,0] + id_accumulator, color='red', s=4) # s=1
                else:
                   plt.scatter(data[:,1], data[:,0] + id_accumulator, color='red', s=4) # s=1  
          id_accumulator = id_accumulator + v2_pop[i].size
   if vector_plot:
      mapped_arrowplot(data_vector, x_dim=scale.v2_pop_size[0], y_dim=scale.v2_pop_size[1], t_max=plot_t_max, t_min=plot_t_min, compatible_output=compatible_output)
elif layer_to_observe == 'v4':
   id_accumulator=0
   data_vector = []
   for i in range(len(v4_pop)):
       data = numpy.asarray(v4_pop[i].getSpikes())
       if vector_plot:
          if plot_time_window:
             data_vector.append(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),:])
          else:
             data_vector.append(data)
       else:
          if len(data) > 0:
             if plot_time_window:
                if compatible_output:
                   plt.scatter(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),1], 
                               data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),0] 
                                   + id_accumulator, 
                               color='blue', s=4) # s=1
                else:
                   plt.scatter(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),1], 
                               data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),0] 
                                   + id_accumulator, 
                               color='blue', s=4) # s=1        
             else:
                if compatible_output:
                   plt.scatter(data[:,1], data[:,0] + id_accumulator, color='blue', s=4) # s=1
                else:
                   plt.scatter(data[:,1], data[:,0] + id_accumulator, color='blue', s=4) # s=1 
          id_accumulator = id_accumulator + v4_pop[i].size
   if vector_plot:
      mapped_arrowplot(data_vector, x_dim=scale.v4_pop_size[0], y_dim=scale.v4_pop_size[1], t_max=plot_t_max, t_min=plot_t_min, compatible_output=compatible_output)
elif layer_to_observe == 'pfc_pre':
   id_accumulator=0
   data_vector = []
   for i in range(len(pfc_pre)):
       data = numpy.asarray(pfc_pre[i].getSpikes())
       if vector_plot:
          if plot_time_window:
             data_vector.append(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),:])
          else:
             data_vector.append(data)
       else:
          if len(data) > 0:
             if plot_time_window:
                if compatible_output:
                   plt.scatter(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),1], 
                               data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),0] 
                                   + id_accumulator, 
                               color='blue', s=4) # s=1
                else:
                   plt.scatter(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),1], 
                               data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),0] 
                                   + id_accumulator, 
                               color='blue', s=4) # s=1        
             else:
                if compatible_output:
                   plt.scatter(data[:,1], data[:,0] + id_accumulator, color='yellow', s=4) # s=1
                else:
                   plt.scatter(data[:,1], data[:,0] + id_accumulator, color='yellow', s=4) # s=1 
          id_accumulator = id_accumulator + pfc[i].size
   if vector_plot:
      mapped_arrowplot(data_vector, x_dim=scale.pfc_pop_size[0], y_dim=scale.pfc_pop_size[1], t_max=plot_t_max, t_min=plot_t_min, compatible_output=compatible_output)
elif layer_to_observe == 'pfc':
   id_accumulator=0
   data_vector = []
   for i in range(len(pfc)):
       data = numpy.asarray(pfc[i].getSpikes())
       if vector_plot:
          if plot_time_window:
             data_vector.append(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),:])
          else:
             data_vector.append(data)
       else:
          if len(data) > 0:
             if plot_time_window:
                if compatible_output:
                   plt.scatter(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),1], 
                               data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),0] 
                                   + id_accumulator, 
                               color='blue', s=4) # s=1
                else:
                   plt.scatter(data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),1], 
                               data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),0] 
                                   + id_accumulator, 
                               color='blue', s=4) # s=1        
             else:
                if compatible_output:
                   plt.scatter(data[:,1], data[:,0] + id_accumulator, color='yellow', s=4) # s=1
                else:
                   plt.scatter(data[:,1], data[:,0] + id_accumulator, color='yellow', s=4) # s=1 
          id_accumulator = id_accumulator + pfc[i].size
   if vector_plot:
      mapped_arrowplot(data_vector, x_dim=scale.pfc_pop_size[0], y_dim=scale.pfc_pop_size[1], t_max=plot_t_max, t_min=plot_t_min, compatible_output=compatible_output)

if layer_to_observe == 'lip' or layer_to_observe == 'all':
   # make a 2D array for plotting (lip)

    print "LIP size is", scale.lip_pop_size[0], 'x', scale.lip_pop_size[1]

    lip_array = numpy.zeros((int(scale.lip_pop_size[0]),int(scale.lip_pop_size[1])),float)

    # Analysis and plotting of LIP spikes

    lip_spikes = lip.getSpikes()

    lip_counts = numpy.zeros((int(scale.lip_pop_size[0])*int(scale.lip_pop_size[1])),int)

    lip_id = []
    lip_times = []
    xvals_lip = []
    yvals_lip = []

    # Do a plot of all spikes
    if compatible_output:
       for sp in lip_spikes:
           lip_id.append(sp[0])
           lip_counts[sp[0]] +=1
           lip_times.append(sp[1])
           xpos = lip_x_array[sp[0]]
           ypos = lip_y_array[sp[0]]
           xvals_lip.append(xpos)
           yvals_lip.append(ypos)
    else:
       for sp in lip_spikes:
           lip_id.append(sp[0])
           lip_counts[sp[0]] +=1
           lip_times.append(sp[1])
           xpos = lip_x_array[sp[0]]
           ypos = lip_y_array[sp[0]]
           xvals_lip.append(xpos)
           yvals_lip.append(ypos)

    print lip_counts

    lip_total = sum(lip_counts)

    print "Total activity", lip_total

    # Get the coordinates of the most active area
    # and do a coarse mapping up to input resolution
    activeLip = numpy.argmax(lip_counts)
    lip_mag = round(float(scale.input_size[0])/float(scale.lip_pop_size[0])) 
    print activeLip, lip_counts[activeLip], lip_x_array[activeLip],lip_y_array[activeLip],lip_x_array[activeLip]*lip_mag, lip_y_array[activeLip]*lip_mag

    x_attend = double(lip_x_array[activeLip]*lip_mag)
    y_attend = double(lip_y_array[activeLip]*lip_mag)
    print "Salient position in Input Space", x_attend, y_attend

    max_activation = float(lip_counts[activeLip])/float(lip_total)

    print "Max Activation", max_activation

    if record_lip_stats or lip_map_on:
       data = numpy.asarray(lip_spikes)

    if record_lip_stats:
       if preferred_orientation == 0: 
          pref_o = "horiz"
       elif preferred_orientation == 1:
          pref_o = "diag"
       elif preferred_orientation == 2:
          pref_o = "vert"
       else:
          pref_o = "ctrdiag"
       if aversive_orientation == 0: 
          avert_o = "horiz"
       elif aversive_orientation == 1:
          avert_o = "diag"
       elif aversive_orientation == 2:
          avert_o = "vert"
       else:
          avert_o = "ctrdiag" 
       try:
          lip_stats = open("./lip_stats_%s_%s_%s_%s_%s_%s.dat" % (base_name.replace("./", ""), pref_o, avert_o, "fb_on" if feedback_on else "fb_off", "FEF_on" if active_pfc else "FEF_off", "learn" if plasticity_on else "static"), "r+")
          old_hist_data = numpy.loadtxt(lip_stats, dtype=int, delimiter=",")
       except IOError:
          lip_stats = open("./lip_stats_%s_%s_%s_%s_%s_%s.dat" % (base_name.replace("./", ""), pref_o, avert_o, "fb_on" if feedback_on else "fb_off", "FEF_on" if active_pfc else "FEF_off", "learn" if plasticity_on else "static"), "w+")
          old_hist_data = None
       if compatible_output:
          h_sort_ids = numpy.lexsort((data[:,1], data[:,0]))  # sort the data by neuron first, then by time of spike.
          hist_data = numpy.asarray(numpy.transpose([numpy.asarray(data[h_sort_ids,0], dtype=int), data[h_sort_ids,1]]), dtype=int)
       else:
          h_sort_ids = numpy.lexsort((data[:,0], data[:,1]))  # sort the data by neuron first, then by time of spike.
          hist_data = numpy.asarray(numpy.transpose([numpy.asarray(data[h_sort_ids,1], dtype=int), data[h_sort_ids,0]]), dtype=int)
       spike_hist = numpy.histogram(hist_data[:,0], bins=numpy.arange((scale.lip_pop_size[0]*scale.lip_pop_size[1])+2))
       spike_hist = numpy.transpose(numpy.asarray([spike_hist[1][:-1], spike_hist[0]], dtype=int))
       if old_hist_data is not None and len(old_hist_data):
          spike_hist[old_hist_data[:,0],1] += old_hist_data[:,1]
       lip_stats.seek(0)
       numpy.savetxt(lip_stats, spike_hist, fmt='%d', delimiter=',')
       lip_stats.close()

    if lip_map_on:
       if plot_time_window:
          if compatible_output:
             data_vector = data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),:]
          else:
             data_vector = data[(data[:,1] >= p_t_w[0]) & (data[:,1] < p_t_w[1]),:]
       else:
          data_vector = data

       mapped_boxplot(data=data_vector, x_dim=scale.lip_pop_size[0], y_dim=scale.lip_pop_size[1], t_max=plot_t_max, t_min=plot_t_min, tau=16, compatible_output=compatible_output)

    else:
       # Plotting of LIP saliency map

       # Calculate LIP map

       for nrn in xrange(scale.lip_pop_size[0]*scale.lip_pop_size[1]):
           xpos = lip_x_array[nrn]
           ypos = lip_y_array[nrn]
           lip_array[xpos,ypos] = float(lip_counts[nrn])/float(lip_total)

       print lip_array

       # make a custom colormap for plotting

       colormap = numpy.array([(0.0,0.0,0.0),
         		       (0.1,0.1,0.1),
			       (0.2,0.2,0.2),
			       (0.3,0.3,0.3), 
			       (0.4,0.4,0.4),        
			       (0.5,0.5,0.5),
			       (0.6,0.6,0.6),
			       (0.7,0.7,0.7),
			       (0.8,0.8,0.8),
			       (0.9,0.9,0.9),
			       (1.0,1.0,1.0)])
								     
       ColMap = ListedColormap(colormap, name='attncolmap')

       register_cmap(cmap=ColMap)

       x = numpy.arange(0,scale.lip_pop_size[0]+1)
       y = numpy.arange(0,scale.lip_pop_size[1]+1)
       X,Y = numpy.meshgrid(x,y)

       plt.figure()
       plt.pcolor(X, Y, lip_array, shading='faceted', cmap=ColMap, vmin=0.0, vmax=max_activation)
       plt.colorbar()
       plt.title("LIP map")

       plt.figure()
       plt.plot(lip_times,lip_id,'.b')
       plt.xlim(0,runtime)
       plt.ylim(0,scale.lip_pop_size[0]*scale.lip_pop_size[1])
       plt.title("LIP spikes")

    if metrics_on:
       met_data = numpy.asarray(lip_spikes)
       actual_objs = visual_metrics.get_annotations(input_file_name=MetricFile)
       rescaled_objs = visual_metrics.scale_annotations(annotations=actual_objs, scale_x=1/(1.6*2.0), scale_y=1/(1.6*2.0))
       biassed_objs = visual_metrics.bias_annotations(annotations=rescaled_objs, preferred=preferred_orientation, aversive=aversive_orientation)
       performance = visual_metrics.attn_performance_monitor(data=met_data, objects=biassed_objs, y_dim=scale.pfc_pop_size[1], t_window=metric_window, t_w_offset=metric_start_offset, t_start=metric_t_start, t_stop=metric_t_stop)
       if output_to_file:
          output_file.write("Metric time window: %d ms\n" % metric_window) 
          output_file.write("Metric window offset %d ms\n" % metric_start_offset)
          output_file.write("Start recording metrics at: %d ms\n" % metric_t_start)
          output_file.write("Stop recording metrics at: %d ms\n" % metric_t_stop)
          output_file.write("--------------------------------------\n")
          output_file.write("PERFORMANCE: \n")
          output_file.write("Time reference    Metric\n")
          output_file.write("__________________________\n")
          t_ref = metric_t_start+metric_start_offset
          for t_m in performance:
              output_file.write("%d ms             %f\n" % (t_ref, t_m))
              t_ref += metric_window
          output_file.write("--------------------------------------------------------------------------------\n\n")
          output_file.close() 
       else:
          print "Computed network performance(s) for this trial: %s\n" % performance
    else:
       if output_to_file: 
          output_file.write("--------------------------------------------------------------------------------\n\n")
          output_file.close() 
if layer_to_observe == 'input_pol_1' or layer_to_observe == 'all':
   # make a 2D array for plotting (input)

   plotting_array = numpy.zeros((int(scale.input_size[0]),int(scale.input_size[1])),int)

   pop_1_spikes = input_pol_1.getSpikes()

   pop_1_id = []
   pop_1_times = []
   xvals_1 = []
   yvals_1 = []

   # indicate areas of input activation
   for sp in pop_1_spikes:
       pop_1_id.append(sp[0])
       pop_1_times.append(sp[1])
       xpos = input_x_array[sp[0]]
       ypos = input_y_array[sp[0]]
       xvals_1.append(xpos)
       yvals_1.append(ypos)
       plotting_array[xpos,ypos] = 3

   # indicate salient position in input space
   # calculated from LIP activity

   x = numpy.arange(0,scale.input_size[0]+1)
   y = numpy.arange(0,scale.input_size[1]+1)
   X,Y = numpy.meshgrid(x,y)

   plt.figure()
   plt.pcolor(X, Y, plotting_array, shading='faceted', cmap=cmap.spectral)
   plt.xlim(0,scale.input_size[0])
   plt.ylim(0,scale.input_size[1])
   plt.title("Input pop 1")

plt.show()

#end()

