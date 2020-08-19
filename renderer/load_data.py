import numpy as np
na = np.newaxis
import os
import h5py

def load_behavior_data(root_folder, num_frames, datatype="scalars", image_width=80, image_height=80):
    """Convenience function for reading in extracted data.
    Example: 
    
    from util import load_behavior_data
    the_folder = "/Users/Groucho/Mousedata/"

    mouse_centroid = load_behavior_data(the_folder, 9000, "centroid")
    x,y = mouse_centroid[:,0], mouse_centroid[:,1]
    plot(x,y,'-k'); axis('equal');
    images = load_behavior_data(the_folder, 9000, "images")
    figure()
    imshow(images[100]); colorbar()
    """
    
    assert datatype in ['scalars', 'images', 'diff_images', \
                        'spines', 'timestamps', \
                        'spines_y', 'spines_z', 'centroid_x', \
                        'centroid_y', 'head_x', 'head_y', \
                        'tail_x', 'tail_y', 'angle', 'height',\
                        'width', 'length', 'area', 'contour', \
                        'centroid', 'tail', 'head', 'velocity', 'velocity_tail']
    img_path = os.path.join(root_folder, 'Extracted Mouse Images.int16binary')
    diff_img_path = os.path.join(root_folder, 'Extracted Mouse Difference Images.int16binary')
    scalar_path = os.path.join(root_folder, 'Scalars (frame, ts, centroid x,y head x,y, tail x,y, angle, height, width, length, area).float32binary')
    spine_path = os.path.join(root_folder, 'Mouse Spine XYZ Interleaved.int32binary')
    ts_path = os.path.join(root_folder, 'Timestamps.int64binary')
    contour_path = os.path.join(root_folder, 'Outline Contour XY Interleaved.int16binary')
    
    num_scalars = 13
    if datatype == 'scalars':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))
    elif datatype == 'images':
        return np.fromfile(img_path, '>i2', image_height*image_width*num_frames).reshape((-1,image_height, image_width))
    elif datatype == 'diff_images':
        return np.fromfile(diff_img_path, '>i2', image_height*image_width*num_frames).reshape((-1,image_height, image_width))
    elif datatype == 'spines':
        return np.fromfile(spine_path, '>i4', image_width*num_frames*3).reshape((-1, image_width*3))
    elif datatype == 'spines_y':
        return np.fromfile(spine_path, '>i4', image_width*num_frames*3).reshape((-1, image_width*3))[:,1::3]
    elif datatype == 'spines_z':
        return np.fromfile(spine_path, '>i4', image_width*num_frames*3).reshape((-1, image_width*3))[:,2::3]
    elif datatype == 'centroid':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,2:4]
    elif datatype == 'centroid_x':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,2]
    elif datatype == 'centroid_y':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,3]
    elif datatype == 'head':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,4:6]
    elif datatype == 'head_x':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,4]
    elif datatype == 'head_y':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,5]
    elif datatype == 'tail':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,6:8]
    elif datatype == 'tail_x':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,6]
    elif datatype == 'tail_y':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,7]
    elif datatype == 'angle':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,8]
    elif datatype == 'height':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,9]
    elif datatype == 'width':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,10]
    elif datatype == 'length':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,11]
    elif datatype == 'area':
        return np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))[:,12]
    elif datatype == 'timestamps':
        return np.fromfile(ts_path, '>i8', num_frames)
    elif datatype == 'velocity_tail':
        data = np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))
        centroid = data[:,2:4]
        tail = data[:,6:8]
        tail_diff = np.sqrt(np.diff(tail, axis=0)**2)
        velocity_axis = np.array([(c-t)/np.linalg.norm(c-t) for (c,t) in zip(centroid, tail)])
        velocity = np.r_[0, [np.dot( t, v.T ) for (t,v) in zip(tail_diff, velocity_axis[1:])]]
        return velocity
    elif datatype == 'velocity':
        data = np.fromfile(scalar_path, '>f4', 13*num_frames).reshape((-1,num_scalars))
        centroid = data[:,2:4]
        velocity = np.r_[0, np.sqrt(np.sum(np.diff(centroid, axis=0)**2, axis=1))]
        return velocity


def load_new_behavior_data(root_folder, num_frames, datatype="scalars"):
    """Convenience function for reading in new extracted data.
    Example:

    from util import load_behavior_data
    the_folder = "/Users/Groucho/Mousedata/"

    mouse_centroid = load_behavior_data(the_folder, 9000, "centroid")
    x,y = mouse_centroid[:,0], mouse_centroid[:,1]
    plot(x,y,'-k'); axis('equal');
    images = load_behavior_data(the_folder, 9000, "images")
    figure()
    imshow(images[100]); colorbar()
    """

    all_data = h5_to_dict(root_folder)
    if datatype == "centroid":
        centroid_x = all_data['scalars']["centroid_x_px"][:num_frames,na]
        centroid_y = all_data['scalars']["centroid_y_px"][:num_frames,na]
        #centroid_x = all_data['scalars']["centroid_x_mm"][:num_frames,na]
        #centroid_y = all_data['scalars']["centroid_y_mm"][:num_frames,na]
        return np.concatenate((centroid_x, centroid_y),axis=1)
    elif datatype == "angle":
        return all_data['scalars'][datatype][:num_frames]
    elif datatype == "images":
        return all_data['frames'][:num_frames,...]



def _load_h5_to_dict(file: h5py.File, path: str) -> dict:
    '''
    Load h5 contents to dictionary.

    Parameters
    ----------
    file (opened h5py File): open h5py File object.
    path (str): path within h5 to dict to load.

    Returns
    -------
    ans (dict): loaded dictionary from h5
    '''

    ans = {}
    if isinstance(file[path], h5py.Dataset):
        # only use the final path key to add to `ans`
        ans[path.split('/')[-1]] = file[path][()]
    else:
        for key, item in file[path].items():
            if isinstance(item, h5py.Dataset):
                ans[key] = item[()]
            elif isinstance(item, h5py.Group):
                ans[key] = _load_h5_to_dict(file, '/'.join([path, key]))
    return ans


def h5_to_dict(h5file, path: str = '/') -> dict:
    '''
    Load h5 dict contents to a dict variable.

    Parameters
    ----------
    h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
    path (str): path to the base dataset within the h5 file. Default: /

    Returns
    -------
    out (dict): dictionary of all h5 contents
    '''

    if isinstance(h5file, str):
        with h5py.File(h5file, 'r') as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, (h5py.File, h5py.Group)):
        out = _load_h5_to_dict(h5file, path)
    else:
        raise Exception('file input not understood - need h5 file path or file object')
    return out
