import itertools
import numpy
import math

"""
This function will translate a 2-D (visual) field's coordinates between
x-y and linear (neuron) mapping. It will handle arbitrary rotations of axis
as well as mirrorings about arbitrarily positioned lines. It will also handle 
any selection of origin in the new axis.

Arguments:

rotation: the angle between the native (neuron) field and the translated
field, expressed in radians
rot_origin_x: the x-coordinate of the translated field's origin, in the native coordinate frame.
rot_origin_y: and likewise for the y-coordinate
dim_x: the width of the field (number of columns)
dim_y: the height of the field (number of rows). Defaults to dim_x.
start_x: the x-coordinate of the point in the original field to start the mapping from
start_y: and likewise for the y-coordinate
end_x: the corresponding x-coordinate to stop the mapping
end_y: and likewise for the y. The mapping will therefore select any window of the entire field to map, as specified by these 4 coordinates.
reflect_x: the line x=reflect_x, *in the translated coordinates* about which to reflect coordinates so that numeration is from right to left.
reflect_y: the line y=reflect_y in the translated coordinates, about which to reflect so that numeration is from top to bottom.

"""

def get_grid_indices(rotation, rot_origin_x, rot_origin_y, dim_x, dim_y=None, start_x=None, start_y=None, end_x=None, end_y=None, reflect_x=None, reflect_y=None, linear_in=False, linear_out=True):
    # instantly return an empty array if x-dimension is 0
    if dim_x == 0:
       return numpy.ma.MaskedArray([])
    # set up defaults
    if dim_y is None:
       dim_y = dim_x
    if start_x is None:
       start_x = 0
    if linear_in:
       start_x = start_x % dim_x
       start_y = start_x / dim_x
       if end_x is not None:
          end_x = end_x % dim_x
          end_y = end_x / dim_x
    elif start_y is None:
       start_y = 0
    if end_x is None:
       end_x = dim_x
    if end_y is None:
       end_y = dim_y
    # set up dimensional transforms to rotate axes
    dir_xform = numpy.array([[numpy.cos(rotation), -numpy.sin(rotation)],[numpy.sin(rotation), numpy.cos(rotation)]])
    dir_rev_xform = numpy.array([[numpy.cos(-rotation), -numpy.sin(-rotation)], [numpy.sin(-rotation), numpy.cos(-rotation)]])
    # compute the absolute maximum bounds - the "sweep range" in the transformed coordinates
    min_x = int(min(rot_origin_x-dim_x+1, -rot_origin_x, math.floor(-math.sqrt(rot_origin_x**2+rot_origin_y**2)), math.floor(1-math.sqrt((rot_origin_x-dim_x)**2+(rot_origin_y-dim_y)**2)))) 
    max_x = int(max(dim_x-rot_origin_x, rot_origin_x+1, math.ceil(1+math.sqrt(rot_origin_x**2+rot_origin_y**2)), math.ceil(math.sqrt((dim_x-rot_origin_x)**2+(dim_y-rot_origin_y)**2)))) 
    min_y = int(min(rot_origin_y-dim_y+1, -rot_origin_y, math.floor(-math.sqrt(rot_origin_x**2+rot_origin_y**2)), math.floor(1-math.sqrt((rot_origin_x-dim_x)**2+(rot_origin_y-dim_y)**2)))) 
    max_y = int(max(dim_y-rot_origin_y, rot_origin_y+1, math.ceil(1+math.sqrt(rot_origin_x**2+rot_origin_y**2)), math.ceil(math.sqrt((dim_x-rot_origin_x)**2+(dim_y-rot_origin_y)**2))))

    """
    next, compute the mask by rotating the original field into the
    transformed coordinate system, offsetting by the start point (lower
    left-hand corner) of the transformed coordinates, then clearing mask
    values for these rotated coordinates. Note the unusual (perverse?) 
    numpy convention where in a masked array a mask value of 1 turns OFF
    the array point (masks out the value). So a 0 in the mask means the 
    array value IS seen.
    """
    mask_indices = numpy.ones((max_x-min_x, max_y-min_y, 2))
    real_indices = numpy.array((numpy.tensordot(numpy.swapaxes(numpy.meshgrid(range(start_x-rot_origin_x, end_x-rot_origin_x), range(start_y-rot_origin_y, end_y-rot_origin_y), indexing='ij'), 0, 2), dir_xform, axes=[[2],[0]]) - [min_x, min_y]), dtype=int)
    mask_apply_indices = numpy.rollaxis(real_indices, 2)
    mask_apply_indices = mask_apply_indices.reshape((mask_apply_indices.shape[0], mask_apply_indices.shape[1]*mask_apply_indices.shape[2]))
    mask_indices[mask_apply_indices[1], mask_apply_indices[0]] = [0, 0] 

    """
    Instantiate the array. This large constructor builds the array of indices
    by iterating over the extended boundaries in the transformed axis, 
    row-by-row considering start and end positions, then converting back to
    the real axis
    """

    xy_indices = numpy.ma.MaskedArray(numpy.array(numpy.tensordot(numpy.swapaxes(numpy.meshgrid(numpy.arange(min_x, max_x), numpy.arange(min_y, max_y), indexing='ij'), 0, 2), dir_rev_xform, axes=[[2],[0]]), dtype=int), mask=mask_indices)
    
    if linear_out:
       # flatten to 1-dimensional indices
       indices = numpy.ma.dot(xy_indices, numpy.array([1, dim_x]))
       return indices
    else:
       # return the x-y indexed array
       return xy_indices

def move_objects(spike_array, x_dim, y_dim, displacements=None):
    if displacements is None or len(displacements) == 0:
       return spike_array
    if 0 not in displacements:
       displacements[0] = (0, 0)
    neuron_ids = numpy.array(spike_array.keys())
    times = numpy.array(spike_array.values())
    data_array = numpy.column_stack([numpy.repeat(neuron_ids[:,numpy.newaxis],times.shape[1], 1).flatten(), times.flatten()])
    displacement_times = numpy.array(displacements.keys())
    displacement_values = numpy.array(displacements.values())
    displacement_indices = numpy.fromiter((numpy.argmax(displacement_times[t >= displacement_times]) for t in numpy.nditer(data_array[:,1])), dtype=int)
    positions_x = data_array[:,0]%y_dim+displacement_values[displacement_indices,0]
    data_array[:,0] = data_array[:,0]+(displacement_values[displacement_indices,0]+displacement_values[displacement_indices,1]*x_dim)
    data_array = data_array[(positions_x >= 0) & (positions_x < x_dim) & (data_array[:,0] > 0) & (data_array[:,0] < (x_dim*y_dim))]
    sort_keys = numpy.lexsort((data_array[:,1], data_array[:,0])) # sort, grouping by neuron ID then time.
    data_array = data_array[sort_keys,:]
    spiking_neurons = itertools.groupby(data_array, lambda x: x[0]) # and taking one group at a time#,
    spikeArray = dict([(neuron[0], numpy.array([spike_time[1] for spike_time in neuron[1]])) for neuron in spiking_neurons]) # create a dictionary indexed by neuron number of the spike times.
    return spikeArray
    
