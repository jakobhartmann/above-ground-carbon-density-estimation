from os.path import exists, join
from os import getcwd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class WaterUtils():
    def __init__(self, dataLoader, water_value):
        self.dataLoder = dataLoader
        self.vegetation_data = self.dataLoder.load_data(vegetation = True)
        self.num_points = self.vegetation_data.shape[0]
        self.water_value = water_value
        self.water_data = self.vegetation_data == self.water_value

    # Get unique vegetation values in dataset
    def get_unique_vegetation_values(self):
        return np.unique(self.vegetation_data)

    # Transform 2d index in 1d index to access reshaped array
    def get_new_index_2d_to_1d(row, col, old_shape):
        return row * (old_shape[0] + 1) + col

    # Generate class map and bitmasks
    def generate_bitmasks(self):
        class_map = np.zeros((self.num_points, self.num_points, self.num_points, self.num_points))
        class_map_dict = {}

        for row in range(class_map.shape[0]):
            y = round(row / 50 - 1, 2)
            for col in range(class_map.shape[1]):
                x = round(col / 50 - 1, 2)
                for row2 in range(class_map.shape[2]):
                    y2 = round(row2 / 50 - 1, 2)
                    for col2 in range(class_map.shape[3]):
                        x2 = round(col2 / 50 - 1, 2)
                        # land x land: 0
                        if (self.water_data[row][col] == False) and (self.water_data[row2][col2] == False):
                            class_map[row][col][row2][col2] = 0
                            class_map_dict[(y, x, y2, x2)] = 0 # data is saved in the form (y, x, y2, x2)!!!
                        # land x water: 1
                        elif self.water_data[row][col] != self.water_data[row2][col2]:
                            class_map[row][col][row2][col2] = 1
                            class_map_dict[(y, x, y2, x2)] = 1
                        # water x water: 2
                        elif (self.water_data[row][col] == True) and (self.water_data[row2][col2] == True):
                            class_map[row][col][row2][col2] = 2
                            class_map_dict[(y, x, y2, x2)] = 2
                        else:
                            print('You messed up!')

        class_map_reshaped = class_map.reshape(self.num_points ** 2, self.num_points ** 2)
        bitmask_land_land = class_map_reshaped == 0
        bitmask_land_water = class_map_reshaped == 1
        bitmask_water_water = class_map_reshaped == 2

        return bitmask_land_land, bitmask_land_water, bitmask_water_water, class_map_dict

    # Returns class map and bitmasks
    def get_bitmasks(self):
        fname1 = self.get_filename('bitmask_land_land')
        fname2 = self.get_filename('bitmask_land_water')
        fname3 = self.get_filename('bitmask_water_water')
        fname4 = self.get_filename('class_map', '.pkl')

        if exists(fname1) and exists(fname2) and exists(fname3) and exists(fname4):
            bitmask_land_land = self.load_bitmask_local(fname1)
            bitmask_land_water = self.load_bitmask_local(fname2)
            bitmask_water_water = self.load_bitmask_local(fname3)
            class_map = self.load_bitmask_local(fname4)
        else:
            bitmask_land_land, bitmask_land_water, bitmask_water_water, class_map = self.generate_bitmasks()
            self.save_bitmask(bitmask_land_land, fname1)
            self.save_bitmask(bitmask_land_water, fname2)
            self.save_bitmask(bitmask_water_water, fname3)
            self.save_bitmask(class_map, fname4)

        return bitmask_land_land, bitmask_land_water, bitmask_water_water, class_map

    # Returns relative location of (saved) class map/bitmask
    def get_filename(self, bitmask_name, suffix = '.npy'):
        fname = "data/" + str(bitmask_name) + '_' + str(self.dataLoder.center_point) + '_' + str(self.dataLoder.num_points) + '_' + str(self.dataLoder.scale) + self.dataLoder.source[-3:] + suffix
        return fname

    # Load class map or bitmask from file
    def load_bitmask_local(self, fname):
        if fname[-4:] == '.npy':
            return np.load(fname)
        elif fname[-4:] == '.pkl':
            with open(fname, 'rb') as f:
                class_map = pickle.load(f)
            return class_map

    # Save class map or bitmask to file
    def save_bitmask(self, file, fname):
        print('Saving within this working directory: ', join(getcwd(), fname))
        if fname[-4:] == '.npy':
            np.save(fname, +file)
        elif fname[-4:] == '.pkl':
            with open(fname, 'wb') as f:
                pickle.dump(file, f)

    # Plot vegetation and water data
    def plot_vegetation_and_water_data(self):
        fig, axes = plt.subplots(1, 2, sharey = True)

        im1 = axes[0].imshow(self.vegetation_data)
        axes[0].set_title('Land cover')
        axes[0].set_xlabel('$x$ (Latitude)')
        axes[0].set_ylabel('$y$ (Longitude)')
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax = cax)

        im2 = axes[1].imshow(self.vegetation_data == self.water_value)
        axes[1].set_title('Permanent water bodies')
        axes[1].set_xlabel('$x$ (Latitude)')
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax = cax)

        return plt