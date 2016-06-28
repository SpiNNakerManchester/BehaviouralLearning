Scripts for the iCub-SpiNNaker visual attention model

This package contains the PyNN scripts needed to run the visual
attention model, on SpiNNaker or alternatively on other simulation
platforms such as NEST (this second option has not been tested!). You
will need Python and PyNN version 0.7 installed: see 
http://neuralensemble.org/trac/PyNN/
for details on PyNN. 

Details for installing the SpiNNaker tool chain are available at
http://spinnakermanchester.github.io/2016.001.AnotherFineProductFromTheNonsenseFactory/

If you do not have a SpiNNaker board you can run the scripts through
the HBP portal at 
http://collaboration.humanbrainproject.eu

List of Files                                
--------------------------------------------------------------------------------
va_network_scaled.py         The top-level file  
gaussiancreatejose.py        Utilities for creating input filters                
vector_topographic_input.py  Output display package to show mapped spikes
spike_file_to_spike_array.py Accept input spikes from a file
visual_network_scaling.py    Scale the network to the input size
transform_mapped_inputs.py   Apply arbitrary translations and rotations to inputs
visual_metrics.py            Simple performance metrics
/data/enhancements           Input data for testing enhanced network
/data/scaling                Input data to try different input scales
/data/learning               Input data for learning experiments

Using the scripts
-----------------
The top-level script file has numerous options and output modes. If
running on SpiNNaker you will need to ensure you have the SpiNNaker 
package imported either similarly to how it is shown:
from pyNN.spiNNaker import *
or:
import pyNN.spiNNaker {as spiNNaker}

Braces indicate an optional renaming; you can give any alias you like
so long as it does not conflict with an existing package you may have
in your local Python install or in this script.

The first form imports all the SpiNNaker-specific names into the local
namespace so that any object instantiation does not need to be
prefaced. If using the second form then objects will need
fully-qualified names: pyNN.spiNNaker.{object} or if using the
aliasing option with as then spiNNaker.{object}

The following options can be set:

layer_to_observe: which of the named layers of the model will be 
selected for spike recording. 'all' = record everything

record_lip_stats: if observing LIP layer output, gather the spike
statistics for each location

lip_map_on: if recording LIP output, map the neuron spike IDs to
topographic locations. Use if output is a topographic plot.

plasticity_on: enable learning
feedback_on: enable inter-layer feedback in the V1-V2-V4 pathway.

aversive_inhibitory: aversive PFC stimulation is applied as 
inhibition to the aversive orientation rather than excitation
to all other orientations.

top_down_priming: directly stimulate V1 from PFC. 

metrics_on: enable performance calculation from some simple metrics.
This requires annotated files with expected attentional position
specified.

vector_plot: Display mapped output spikes for V1/V2/V4 as arrow plots
indicating direction

active_pfc: enable FEF layer

multistage_pfc: enable FEF with hardwired PFC

output_to_file: output recorded spikes to a file for later processing

randomise_spike_times: apply a small jitter to input spike times

compatible_output: if true, spikes are recorded as (id, time) tuples

preferred_orientation: sets the orientation preference. 
0 = horizontal, 1 = pi/4 diagonal, 2 = vertical, 3 = 3 pi/4 diagonal
 
aversive_orientation: sets which orientation is aversive 

base_num_neurons: input size, expressed as the length on a side. So
for example setting 32 will give a 32x32 input grid.

subsample_step: downsample input spike rates by averaging spike counts
over a window of this number of milliseconds. If more than half of the
milliseconds within this window have a spike the subsample will
likewise have a spike.

random_spike_jitter millisecond variability on spike timing if timings
are randomised

input_file_type: Input from: 
0 - a simple list of ids; 
1 - a single full list of spikes and times;
2 - a pair of polarities with spikes and times.

The files 'spikes_in_32x32', 'horiz_right_32x32' and
'run1_32x32_pol1.dat', 'run1_32x32_pol2.dat' give 
examples of each file type
 
metrics_with_input: if using the visual_metrics module, indicates
the annotations are embedded in the input file rather than 
as a separate file (e.g. 'object_annots_48x48')

input_file_base_names: base name(s) of the input file(s). Each
file should be an element in the list 

stimulus_onset_offsets: this list of lists indicates when to apply
the stimuli from each listed input file. Each inner list is associated
with an input file in the order given in input_file_base_names. 
This inner list is a series of tuples indicating the time (in ms) of
onset and offset time for each input file. Multiple onset and offset
times per input file may be specified.
  
compatible_input: if true, input file order is in the format (time, spike)

moving_objects: applies virtual motion to objects (NOT USED)
random_motions: randomise virtual motion (NOT USED)
motion_period: mean time between object motions (NOT USED)
motion_time_jitter: random motion time variablity (NOT USED)
virtual_velocity: if objects are moving, the motion velocity (NOT USED)

save_weights: save output weights after learning to a file

Also look at the section #PARAMETERS NETWORK for all the various
network parameters. We would encourage users to experiment with 
various settings to see how these parameters interact. In general
experimenting with PFC_bias and the weight initialisations will 
tend to produce the most obvious effects. 

gaussiancreatejose.py
---------------------
File 'gaussiancreatejose.py' can create a wide variety of different
input filters depending on the setup. A filter can be created by 
creating some form of ConnectorList (Gaussian, Gabor, or MexicanHat)
in either tuned (gain and eccentricity set by a user parameter) or
untuned (fixed tuning curve) variants and then passing this to a 
Filter2dConnector constructor to generate the connector list necessary
to generate the appropriate filter connection. This list can then 
be used for a FromListConnector. In the top-level script can be 
seen an example of how this is done. 'Tuned' versions have the
following parameters:

scales: How many different scales of filter you wish to use. Each 
filter scale looks at a larger subsample window of the entire input.

orientations: How many orientations the filters will have. If
eccentricity is 0 this has no effect.

sizeg1: the aperture size of the filter: how wide the filter
distributions are at any given value

gain: filter gain - this sets the effective sharpness of the filter
as well as the peak height

eccentricity: - how eccentric the ellipse is of the 2-d filter
response. A high eccentricity will create very 'cigar-shaped'
filters with strong orientation, an eccentricity of 0 will
create an unoriented circular filter.

A Filter2DConnector creates a connection between 2 layers. It
has the following parameters:

size_in1, size_in2: Dimensions of the presynaptic layer

size_out1, size_out2: Dimensions of the postsynaptic layer

weights: a list of the desired weight for each connection. This
is where to plug in the filter connections above.

size_k1, size_k2: absolute dimensions of the filter window. This
is different from the size_g1 parameter in that it sets a hard
cutoff for the connection distribution whereas the size_g1 adjusts
the overall shape of the distribution.

jump: how far to move in each dimension when stepping the filter
window across the input.

delays: a similar list to the weights values used to set input delays.

gain: like an input preamplifier, scales each weight by some fixed
value.

visual_network_scaling.py
-------------------------
Many of these parameters can be automatically set with the
visual_network_scaling.py module. This module provides a means of
scaling the network for a wide range of different inter-layer scales.
For this module, the main object: scale_factor has the following
parameters: 

dimensions: dimensionality of the input (usually will be 2)

dim_sizes: manually set each dimensional extent of the input.
This parameter will expect a list of length dimensions  with an
entry for each dimension, or None if the dimensional extent is
to be automatically broadcast from base_dim_size.

base_dim_size: 'normal' dimensional size in neurons along an input
axis.

subsample_factor_x_y: reduction (or expansion) in size between layer
x and layer y. This applies on a dimension-by-dimension basis. So if
you have an input with 32x32 neurons and an output with 10x10 a
subsample_factor of 3.2 would be appropriate.

base_filter_size: base width, referenced to base_dim_size, of the
input-filter distributions (the sizeg1 parameter)

sweep_filter: create a range of scaled filters, set by the 
filter_scale parameter. 

filter_scale: sets the scale multipliers for the filters. Filters
will be scaled by multiplying the each scale multiplier by 
base_filter_size and then scaling for the relative size of the
layer to be filtered, relative to base_dim_size. This parameter 
can either be an integer (in which case it will generate a list of
scales ranging from 0 to filter_scale_factors-1) or a
tuple (start, stop, step) which operates in the standard Python way.

xy_kernel_quant_cutoff: sets the size_k1 and size_k2 boundaries for a
Filter2DConnector. These will be scaled to the appropriate fixed
window size for the network given the base_dim_size.

base_v2_sweep_subfield: sweeps the pooling sizes of V2 much like
sweep_filter above.

subfield_scale: sets the scale factors for V2 pooling just like
filter_scale.

active_P: Set this if the network has the active_pfc option enabled.

spike_file_to_spike_array.py
----------------------------
spike_file_to_spike_array.py not only permits conversion of spike
files into arrays suitable for a PyNN SpikeSourceArray object, but
can also apply various transformations on these files to generate
various types of input. It has the following functions:

convert_spike_list_to_timed_spikes: changes a simple list of neuron
ID's into a dictionary of spike times indexed by their ID. The user
can specify the start and stop times, and how often the neurons should
spike.

convert_file_to_spikes: converts a PyNN spike file, written in the
as (id, time) if compatible_input is set, (time, id) otherwise, 
into a dictionary similar to the above. 

loop_array: takes a SpikeSourceArray and loops it back to the 
beginning so the same stimulus can be run multiple times. 
Fractional portions of the array can be specified. Parameter
runtime allows the time duration of the looped array to be set
as an absolute value (which will cut off repeats that pass this
limit). Parameter sampletime sets the amount of time of the
original array to use in looping.

splice_arrays, splice_files: allow the user to concatenate 2 or
more arrays or files into a single merged spike array. The user
can specify both which neurons are to be used in the merged array
and which times to use from each array/file. The input_times
parameter expects a list of (onset, offset) pairs just like
stimulus_onset_offsets in the top-level sctipt. input_neurons
should be a list of lists of ID's for each array or file.

subsample_spikes_by_time: takes a SpikeSourceArray and subsamples
to a coarser resolution by locally averaging the number of spikes
in a given subsample window and outputting a spike if the number is
more than half the time of the window.

random_skew_times: takes a SpikeSourceArray and randomly perturbs
each of the spike times by an amount drawn from a uniform
distribution. The width of the perturbation window can be set via
parameter skewtime.

generate_shadow_spikes: creates a second set of spikes offset in 
time and position from the originals - as if the 'shadow' of an
object has passed where an object given a certain virtual motion
was.  

vector_topographic_activity_plot.py
-----------------------------------
This module enables the generation of the mapped output diagrams for
network activity. You can experiment with different colour maps for
both box-plots and arrow-plots to get best visibilty. The following
plot types are available:

mapped_boxplot: creates a series of boxes each mapped to a topographic
location in the output space. Each box represents a spike, larger
boxes represent earlier spikes. A leaky integrator colours the boxes
to indicate time-averaged activity rates. Make sure to set screen_size
(in diagonal inches) screen_aspect (hxw) and screen_v_resolution to 
match your display. max_v_screen occupancy indicates how much of the 
total vertical dimension of your screen the plot is allowed to take.
t_max and t_min indicate the stop and start points in the recorded 
spikes to use in generating the plot. This is independent of the run
length and stimulus onset/offset times. tau is the time constant on 
the leaky integrator in ms. x_dim and y_dim give the actual size of
the displayed grid in boxes.

mapped_arrowplot: creates a series of arrow-plots for
orientation-specific layers. Similar parameters to mapped_boxplot. 
 
