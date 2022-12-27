from conversions import m_to_deg
from visualization import heatmap_comparison
import ee
import numpy as np
from os.path import exists

class DataLoad:
    def __init__(self, source, center_point, num_points, scale, veg_idx_band, data_load_type):
        print("2. Initialize the new instance of Point.")
        self.source = source
        self.center_point = center_point
        self.num_points = num_points
        self.scale = scale
        self.veg_idx_band = veg_idx_band
        self.data_load_type = data_load_type

    # load data depending on the chosen type
    def load_data(self):
        if self.data_load_type == 'api':
            return self.load_data_api()
        if self.data_load_type == 'local':
            return self.load_data_local()
        if self.data_load_type == 'optimal':
            # load from filesystem if exists
            if exists(self.get_filename()):
                return self.load_data_local()
            else: 
                # calls api and saves data to local file
                self.load_data_api()
                self.save_data()
                return self.dataset
        
        print("Data load type not found")
        assert(0)

    # Load data from api
    def load_data_api(self,):
        # Trigger the authentication flow. Can comment out if auth token cached, eg after running it once
        # ee.Authenticate(auth_mode="notebook")
        # Initialize the library.
        ee.Initialize()
        source_dataset = ee.ImageCollection(self.source)

        # Setup the domain of our estimation
        x_space = np.linspace(-1, 1, self.num_points)
        y_space = np.linspace(-1, 1, self.num_points)
        xy_space = np.meshgrid(x_space, y_space)
        xy_space = np.stack((xy_space[1], xy_space[0]), axis=-1).reshape(-1, 2)

        # Make coordinates for sampling api
        coords = self.center_point + m_to_deg(xy_space * (self.scale * self.num_points) / 2)

        # Sampled Dataset
        dataset = np.array([])

        # Iteratively call api with 5000 points
        coords_copy = coords.copy()
        while len(coords_copy)>0:

            # Make 5000 batch
            max_index = min(len(coords_copy), 5000)
            buffer = coords_copy[0:max_index]
            if max_index == len(coords_copy):
                coords_copy = []
            else:
                coords_copy = coords_copy[max_index:]
            
            # Call API
            geom_coords = ee.FeatureCollection(
            [ee.Geometry.Point(c[0], c[1]) for c in buffer])
            samples = source_dataset.mean().reduceRegions(**{
                'collection': geom_coords,
                'scale': self.scale,
                'reducer': 'mean'}).getInfo()
            for sample in samples['features']:
                dataset = np.append(dataset, [sample['properties'][self.veg_idx_band]])
        self.dataset = dataset.reshape([self.num_points, self.num_points])
        return self.dataset

    # returns location of saved dataset
    def get_filename(self):
        fname = "data/"+'data_'+str(self.center_point)+'_'+str(self.num_points)+'_'+str(self.scale)+self.source[-3:]+'.npy'
        return fname
    
    # load data from filesystem
    def load_data_local(self):
        if hasattr(self, 'dataset'):
            return self.dataset

        # generate filename
        fname = self.get_filename()

        # load dataset from file system
        self.dataset = np.load(fname)
        return self.dataset
    
    # save data to filesystem
    def save_data(self):
        assert(hasattr(self, 'dataset'))
        fname = self.get_filename()
        np.save(fname, +self.dataset)
    
    # Returns values for given points, normalized coordinates (-1, 1)
    def load_values(self, normal_indices):
        indices = ((normal_indices+1.0)/2*(self.num_points-1)).astype(np.int_)
        if len(indices)==0:
            return np.array([])
        else:
            return np.array([np.array([self.dataset[v[0],v[1]]]) for v in indices])






