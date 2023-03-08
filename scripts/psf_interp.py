import datetime
from datetime import date
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy import units as u
import scopesim as sim
import scopesim_templates as sim_tp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import cv2
import astropy
from astropy.io import fits
from astropy.table import Table

#0.004  is resolution 4mas/pixel for wide FoV of MICADO
today = date.today()

def generate_cluster(wmin, wmax, pix_scale = 4*10**(-3)):
    cluster = sim_tp.cluster(mass=10000, core_radius=0.5, distance=8000)
    x_cluster = cluster.fields[0]["x"].data/0.004
    y_cluster = cluster.fields[0]["y"].data/0.004
    #fluxes_cluster = cluster.fluxes(wave_min=wmin, wave_max=wmax)#, pixel_scale=4*u.arcsec*10**(-3)).image
    #print(fluxes_cluster, type(fluxes_cluster.data))
    image_cluster = cluster.image(wave_min=wmin, wave_max=wmax, pixel_scale=pix_scale*u.arcsec).image
#     fig = plt.figure(figsize=(20,20))
#     ax = plt.subplot(1,1,1)
#     ax.imshow(image_cluster, vmin=0, vmax=1, cmap='PuRd')
    #X, Y = np.meshgrid(x_cluster, y_cluster)
    
    #plt.pcolormesh(X, Y, fluxes_cluster, shading='auto')
    #plt.show()
    return x_cluster, y_cluster, image_cluster

def pad_image_to_box(image, n): #n == 3 for  grid 3x3
    
    """
    Zero-padding to box 
    Added zeros to array for smaller sides
    
    image: np.array of np.float64
    
    n: int
    
    Size of psfs grid
    (For example: if grid 3x3, n == 3)
    
    """
    
    y_size = image.shape[0]
    x_size = image.shape[1]
    resid = abs(y_size - x_size) #want to make box
    if y_size > x_size:
        padded_img = np.pad(image, ((0, n-y_size%n),(resid, n-y_size%n)), 'constant',constant_values=((0, 0),(0, 0)))
        return padded_img
    else:
        padded_img = np.pad(image, ((resid, x_size%n),(0, x_size%n)), 'constant',constant_values=((0, 0),(0, 0)))
        return padded_img
        
    
def resize_psf_grid(psf_grid_table):
    x = psf_grid_table['x'].data/0.004
    y = psf_grid_table['y'].data/0.004
    X, Y = np.meshgrid(x, y)
    plt.figure(figsize=(20,20))
    plt.title('X')
    plt.imshow(X)
    plt.show()
    plt.figure(figsize=(20,20))
    plt.title('Y')
    plt.imshow(Y)
    plt.show()
    return X,Y
    
def interp_weights(psfs_centres, z, x_min, x_max, y_min, y_max):
    
    """
    Compute and return linear interpolation weights
    Plot linear interpolation weights
    
    psfs_centres: list of tupels (x_i, y_i)
    
    Coordinates of centers psf, 
    which situated in cropped overlapped images
    
    z: list of np.float64 
    
    Meaning in each center coordinate
    len(z)==len(psfs_centres)
    
    x_min, x_max, y_min, y_max: int
    
    Bounds of interpolation
    
    """

    X = range(x_min, x_max)#np.linspace(

    Y = range(y_min, y_max)#np.linspace

    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation

    interp = LinearNDInterpolator(psfs_centres, z)
    #print(list(zip(x, y)))
    print(z)

    Z = interp(X, Y)
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(20,20))
    
    plt.pcolormesh(X, Y, Z, shading='auto')

    plt.plot([i[0] for i in psfs_centres], [i[1] for i in psfs_centres], "ok", label="input point")

    plt.legend()

    plt.colorbar()

    plt.axis("equal")
    plt.savefig(f'linear_interp{today}.png')
    plt.show()
    print(X.shape,Y.shape)
    print(Z)
    print(Z.shape)
    return Z

def chose_psf_equal_grid(hdul, row_idx, column_idx, wave_id=2):
    grid = hdul[wave_id].data
#     psfs = [[grid[44],
#                 grid[43],
#                 grid[42]],
#                 [grid[45],
#                 grid[0],
#                 grid[41]],
#                 [grid[46],
#                 grid[47],
#                 grid[48]]
#            ]
    key_idx = '{}{}'.format(row_idx, column_idx)
    psfs = {'00': [grid[44], grid[43], grid[45], grid[0]],
           '01': [grid[43], grid[44], grid[42], grid[45], grid[0], grid[41]],
           '02': [grid[42], grid[43], grid[0], grid[41]],
           '10': [grid[45], grid[0], grid[46], grid[47]],
           '11': [grid[0], grid[44], grid[43], grid[42], grid[41], grid[45], grid[46], grid[47], grid[48]],
           '12': [grid[0], grid[41], grid[47], grid[48]],
           '20': [grid[46], grid[45], grid[0], grid[47]],
           '21': [grid[47], grid[0], grid[45], grid[46], grid[41], grid[48]],
           '22': [grid[48], grid[0], grid[41], grid[47]]}
    #(24,24)(48,24)(24,48)(0,24)(24,0)(0,0)(48,48)(0,48)(48,0)
    zs = {'00': [np.array([1,0,0,0,0,0,0,0,0]),
                np.array([0,1,0,0,0,0,0,0,0]),
                np.array([0,0,1,0,0,0,0,0,0]),
                np.array([0,0,0,0,0,0,1,0,0])],
         '01': [np.array([1,0,0,0,0,0,0,0,0]),
               np.array([0,0,0,1,0,0,0,0,0]),
               np.array([0,1,0,0,0,0,0,0,0]),
               np.array([0,0,0,0,0,0,0,1,0]),
               np.array([0,0,1,0,0,0,0,0,0]),
               np.array([0,0,0,0,0,0,1,0,0])],
         '02': [np.array([1,0,0,0,0,0,0,0,0]),
               np.array([0,0,0,1,0,0,0,0,0]),
               np.array([0,0,0,0,0,0,0,1,0]),
               np.array([0,0,1,0,0,0,0,0,0])],
         '10': [np.array([1,0,0,0,0,0,0,0,0]),
               np.array([0,1,0,0,0,0,0,0,0]),
               np.array([0,0,1,0,0,0,0,0,0]),
               np.array([0,0,0,0,0,0,1,0,0])],
         '11': [np.array([1,0,0,0,0,0,0,0,0]),
               np.array([0,0,0,0,0,1,0,0,0]),
               np.array([0,0,0,0,1,0,0,0,0]),
               np.array([0,0,0,0,0,0,0,0,1]),
               np.array([0,1,0,0,0,0,0,0,0]),
               np.array([0,0,0,1,0,0,0,0,0]),
               np.array([0,0,0,0,0,0,0,1,0]),
               np.array([0,0,1,0,0,0,0,0,0]),
               np.array([0,0,0,0,0,0,1,0,0])
               ],
         '12': [np.array([0,0,0,1,0,0,0,0,0]),
               np.array([1,0,0,0,0,0,0,0,0]),
               np.array([0,0,0,0,0,0,0,1,0]),
               np.array([0,0,1,0,0,0,0,0,0])],
         '20': [np.array([1,0,0,0,0,0,0,0,0]),
               np.array([0,0,0,0,1,0,0,0,0]),
               np.array([0,0,0,0,0,0,0,0,1]),
               np.array([0,1,0,0,0,0,0,0,0]),
               ],
         '21': [np.array([1,0,0,0,0,0,0,0,0]),
               np.array([0,0,0,0,1,0,0,0,0]),
               np.array([0,0,0,0,0,1,0,0,0]),
               np.array([0,0,0,1,0,0,0,0,0]),
               np.array([0,0,0,0,0,0,0,0,1]),
               np.array([0,1,0,0,0,0,0,0,0])],
         '22': [np.array([1,0,0,0,0,0,0,0,0]),
               np.array([0,0,0,0,0,1,0,0,0]),
               np.array([0,0,0,0,1,0,0,0,0]),
               np.array([0,0,0,1,0,0,0,0,0])]}
    
    return psfs[key_idx], zs[key_idx]

def crop_image_and_blur(image, n, hdul, psf_size=5, weight_matrix='linear'):
    
    """
    Crop image to overlapped cutouts with equal steps
    Use if psf grid have equal steps
    
    image: np.array
    
    n: int
    
    Size of psfs grid
    (For example: if grid 3x3, n == 3)
    
    psf_size: int
    
    How much pixels take from the center psf 
    (For example: if psf_size = 2, psf is 4x4)
    
    hdul: hdul of fits file
    
    """
    
    #psf_size=5#5
    n_cropped = n #+ (n-1) #crop one side of image
    image_size_y = image.shape[0]
    image_size_x = image.shape[1]
    step_y = int(image_size_y/(n+1))#(2*n))
    step_x = int(image_size_x/(n+1))#(2*n))
    crop_size_y = int(2*image_size_y/(n+1))
    crop_size_x = int(2*image_size_x/(n+1))

    fig0 = plt.figure(figsize=(20,20))
    idx_graph=0
    blurred = np.zeros((image_size_y+psf_size*2 -1, image_size_x+psf_size*2 - 1))
    for _,row in enumerate(range(n_cropped)):
        for __,column in enumerate(range(n_cropped)):
            crop_img = image[row*step_y:(row*step_y+crop_size_y), column*step_x:(column*step_x+crop_size_x)]
            psfs,zs = chose_psf_equal_grid(hdul, _, __)
            i = 0
            weights_sum = np.zeros(crop_img.shape)
            for psf, z in zip(psfs, zs): # i enum
                i += 1
                blurred_tmp = np.zeros((image_size_y, image_size_x))
                #print(crop_img.shape)
                if weight_matrix == 'random':
                    weights = np.random.rand(crop_img.shape[0], crop_img.shape[1])

                if weight_matrix == 'circle':
                    step_in_circle = crop_img.shape[0]//2
                    circle   = [np.sqrt(step_in_circle**2-x**2) for x in np.arange(-step_in_circle,
                                                                                        step_in_circle)]
                    weights_diag = np.diagflat(circle)
                    norm_c = np.max(weights_diag)-np.min(weights_diag)
                    weights_diag = weights_diag - np.min(weights_diag)
                    weights_diag=(weights_diag/norm_c)#*(np.max(crop_img)-np.min(crop_img))
                    #print(weights_diag)
                    crop_img = crop_img @ weights_diag #@ weights_diag #weights_rand#weights[:48,:48]
                    plt.figure(figsize=(20,20))  
                    plt.title('After_weighting')#weights[:48,:48]
                    plt.imshow(crop_img, vmin=0, vmax=50, cmap='PuRd')
                    plt.show()
                    

                if weight_matrix == 'linear':
                    
#                     z = np.array([])#np.concatenate((np.array([1]), np.zeros(8)))
#                     norm = np.max(z)-np.min(z)
#                     z = z - np.min(z)
#                     z=z/norm
                    #print(z)
                    print((column*step_x + step_x,row*step_y+step_y))
                    #(24,24)(48,24)(24,48)(0,24)(24,0)(0,0)(48,48)(0,48)(48,0)
                    weights = interp_weights([
                                    (column*step_x + step_x,row*step_y+step_y), 
                                    ((column*step_x+crop_size_x),row*step_y+step_y), 
                                    (column*step_x + step_x,(row*step_y+crop_size_y)),
                                              (column*step_x, row*step_y + step_y),
                                              (column*step_x + step_y,row*step_y),
                                    (column*step_x, row*step_y),
                                    ((column*step_x+crop_size_x),(row*step_y+crop_size_y)),
                                    (column*step_x,(row*step_y+crop_size_y)),
                                    ((column*step_x+crop_size_x),row*step_y)],
                                   z,
                                   column*step_x, (column*step_x+crop_size_x), row*step_y, (row*step_y+crop_size_y))
                    weights_sum += weights#/np.linalg.norm(weights)
                    #print(weights)
                    #print(weights.shape)
                    print(weights.shape)
                    print(weights[:crop_size_y,:crop_size_x].shape, crop_img.shape, crop_size_y, crop_size_x)
                    crop_img = crop_img * weights[:crop_size_y,:crop_size_x]
                    plt.figure(figsize=(20,20))  #np.dot(
                    plt.title('After_weighting')#weights[:48,:48]
                    plt.imshow(crop_img, vmin=0, vmax=50, cmap='PuRd')
                    plt.show()

                plt.figure(figsize=(20,20))
                plt.title('Cropped_image')
                plt.imshow(crop_img, vmin=0,vmax=50, cmap='PuRd')
                plt.show()

                    
                #psf = hdul[2].data[48]
                blurred_tmp[row*step_y:(row*step_y+crop_size_y), column*step_x:(column*step_x+crop_size_x)] += crop_img
                conv_with_psf = scipy.signal.fftconvolve(blurred_tmp, psf[psf.shape[0]//2-psf_size:psf.shape[0]//2+psf_size,
                                                                      psf.shape[1]//2-psf_size:psf.shape[1]//2+psf_size])
                
                plt.figure(figsize=(20,20))
                plt.title('After_convolving')
                plt.imshow(conv_with_psf, vmin=0,vmax=50, cmap='PuRd')
                plt.show()

                plt.figure(figsize=(20,20))
                plt.title('PSF')
                plt.imshow(psf[psf.shape[0]//2-psf_size:psf.shape[0]//2+psf_size,
                               psf.shape[1]//2-psf_size:psf.shape[1]//2+psf_size],
                           norm=LogNorm(),cmap='PuRd')
                plt.show()


#                 conv_with_psf = pad_image_to_box(conv_with_psf, 2)
#                 resid_x = conv_with_psf.shape[1] - crop_img.shape[1]
#                 resid_y = conv_with_psf.shape[0] - crop_img.shape[0]
#                 print(resid_x, resid_y)

#                 idx_y_0 = row*step_y-int(resid_y/2)
#                 add_y_0 = 0
#                 if (idx_y_0 < 0):
#                     idx_y_0 = 0
#                     add_y_0 = psf_size


#                 idx_y_1 = (row*step_y+crop_size_y)+int(resid_y/2)
#                 add_y_1 = conv_with_psf.shape[0]
#                 if (idx_y_1 > image_size_y):
#                     idx_y_1 = image_size_y
#                     add_y_1 = conv_with_psf.shape[0] - psf_size

#                 idx_x_0 = column*step_x-int(resid_x/2)
#                 add_x_0 = 0
#                 if (idx_x_0 < 0):
#                     idx_x_0 = 0
#                     add_x_0 = psf_size

#                 idx_x_1 = (column*step_x+crop_size_x)+int(resid_x/2)
#                 add_x_1 = conv_with_psf.shape[1]
#                 if (idx_x_1 > image_size_x):
#                     idx_x_1 = image_size_x
#                     add_x_1 = conv_with_psf.shape[1] - psf_size

                #print(idx_y_0, idx_y_1, idx_x_0, idx_x_1)

#                 blurred[idx_y_0:idx_y_1, idx_x_0:idx_x_1] += conv_with_psf[add_y_0:add_y_1,add_x_0:add_x_1]
                print('blurred.shape =  ', blurred.shape, 'conv.shape = ', conv_with_psf.shape)
                blurred += conv_with_psf#blurred_tmp

                plt.figure(figsize=(20,20))
                plt.title('PSF_{} row_{} column_{}'.format(i, _, __))
                plt.imshow(blurred, vmin=0, vmax=50, cmap='PuRd')
                plt.show()
            #blurred[row*step_y:(row*step_y+crop_size_y), column*step_x:(column*step_x+crop_size_x)] /= weights_sum
    plt.figure(figsize=(20,20))
    plt.title('Blurred_final')
    plt.imshow(blurred, vmin=0, vmax=50, cmap='PuRd')
    plt.savefig('result_blurred_{}_{}.png'.format(today, weight_matrix))
    plt.show()
    return blurred

def crop_image_and_blur_radial(image, n, hdul):
    
    """
    Crop image to overlapped cutouts
    Use if psf grid have not equal complex psf grid
    
    image: np.array
    
    n: int
    
    Size of psfs grid
    (For example: if grid 3x3, n == 3)
    
    psf_size: int
    
    How much pixels take from the center psf 
    (For example: if psf_size = 2, psf is 4x4)
    
    hdul: hdul of fits file
    
    """
    
    
    n_cropped = n #+ (n-1) #crop one side of image
    image_size_y = image.shape[0]
    image_size_x = image.shape[1]
    step_y = int(image_size_y/(n+1))#(2*n))
    step_x = int(image_size_x/(n+1))#(2*n))
    crop_size_y = int(2*image_size_y/(n+1))
    crop_size_x = int(2*image_size_x/(n+1))

    fig0 = plt.figure(figsize=(20,20))
    idx_graph=0
    blurred = np.zeros((image_size_y+psf_size, image_size_x+psf_size))
    for _,row in enumerate(range(n_cropped),1):
        for __,column in enumerate(range(n_cropped),1):
            crop_img = image[row*step_y:(row*step_y+crop_size_y), column*step_x:(column*step_x+crop_size_x)]
            print(crop_img.shape)
            weights_rand = np.random.rand(crop_img.shape[0], crop_img.shape[1])
            step_in_circle = crop_img.shape[0]//2
            circle   = [np.sqrt(step_in_circle**2-x**2) for x in np.arange(-step_in_circle,
                                                                                step_in_circle)]
            weights_diag = np.diagflat(circle)
            norm_c = np.max(weights_diag)-np.min(weights_diag)
            weights_diag = weights_diag - np.min(weights_diag)
            weights_diag=(weights_diag/norm_c)#*(np.max(crop_img)-np.min(crop_img))
            #print(weights_diag)
            z=[8, 10, 6,0,0,  0, 0]
            norm = np.max(z)-np.min(z)
            z = z - np.min(z)
            z=z/norm
            #print(z)
            weights = interp_weights([(48,24),
                            (24,24),
                            (24,48),
                            (0,0),
                            (48,48),
                            (0,48),
                            (48,0)],
                           z,
                           0, 48, 0, 48)
            print(weights)
            print(weights.shape)
            
            plt.figure(figsize=(20,20))
            plt.title('Cropped_image')
            plt.imshow(crop_img, vmin=0,vmax=50, cmap='PuRd')
            plt.show()
            
            crop_img = crop_img @ weights_diag#@ weights_diag #weights_rand#weights[:48,:48]
            
            plt.figure(figsize=(20,20))  #np.dot(
            plt.title('After_weighting')#weights[:48,:48]
            plt.imshow(crop_img @ weights_diag, vmin=0, vmax=50, cmap='PuRd')
            plt.show()
            
            
            psf = hdul[2].data[48]
            conv_with_psf = scipy.signal.fftconvolve(crop_img, psf[len(psf)//2-psf_size:len(psf)//2+psf_size,
                                                                  len(psf)//2-psf_size:len(psf)//2+psf_size])
            plt.figure(figsize=(20,20))
            plt.title('After_convolving')
            plt.imshow(conv_with_psf, vmin=0,vmax=50, cmap='PuRd')
            plt.show()
            
            plt.figure(figsize=(20,20))
            plt.title('PSF')
            plt.imshow(psf[psf.shape[0]//2-psf_size:psf.shape[0]//2+psf_size,
                           psf.shape[1]//2-psf_size:psf.shape[1]//2+psf_size],
                       norm=LogNorm(),cmap='PuRd')
            plt.show()
            

            conv_with_psf = pad_image_to_box(conv_with_psf, 2)
            resid_x = conv_with_psf.shape[1] - crop_img.shape[1]
            resid_y = conv_with_psf.shape[0] - crop_img.shape[0]
            print(resid_x, resid_y)
            
            idx_y_0 = row*step_y-int(resid_y/2)
            add_y_0 = 0
            if (idx_y_0 < 0):
                idx_y_0 = 0
                add_y_0 = psf_size
                
                
            idx_y_1 = (row*step_y+crop_size_y)+int(resid_y/2)
            add_y_1 = conv_with_psf.shape[0]
            if (idx_y_1 > image_size_y):
                idx_y_1 = image_size_y
                add_y_1 = conv_with_psf.shape[0] - psf_size
                
            idx_x_0 = column*step_x-int(resid_x/2)
            add_x_0 = 0
            if (idx_x_0 < 0):
                idx_x_0 = 0
                add_x_0 = psf_size
                
            idx_x_1 = (column*step_x+crop_size_x)+int(resid_x/2)
            add_x_1 = conv_with_psf.shape[1]
            if (idx_x_1 > image_size_x):
                idx_x_1 = image_size_x
                add_x_1 = conv_with_psf.shape[1] - psf_size
                
            print(idx_y_0, idx_y_1, idx_x_0, idx_x_1)
            
            blurred[idx_y_0:idx_y_1, idx_x_0:idx_x_1] += conv_with_psf[add_y_0:add_y_1,add_x_0:add_x_1]

                
            plt.figure(figsize=(20,20))
            plt.imshow(blurred, vmin=0, vmax=50, cmap='PuRd')
            plt.savefig('result_blurred.png')
            plt.show()
    return blurred
            
def plot_psfs(hdul, save = True):
    
    """
    Plot psfs grid on a FoV (?)
    Plot 48 psfs (left upper 0th, right bottom  48th)
    
    hdul: fits file hdul
    
    j: int
    
    Number of extenshion of 
    fits file "AnisoCADO_SCAO_FVPSF_4mas_EsoMedian_20190328.fits"
    
    """
    
    plt.figure(figsize=(25,25))
    #plt.rcParams.update({'font.size': 15})
    plt.title('PSF Grid', size=45)
    t = Table.read(hdul[1], format='fits')
    plt.scatter(t['x'], t['y'], 100*np.ones(len(t['x'])))
    for i, x, y in zip(range(len(t['x'])), t['x'], t['y']):
        plt.annotate(i, (x, y),size=15)
    #plt.rcParams.update({'font.size': 35})
    if save:
        plt.savefig("PSF_GRIG_ANISOCADO.pdf")
    

    for k in range(2,7):
        plt.figure(figsize=(55,55))
        plt.rcParams.update({'font.size': 25})
        plt.suptitle('PSF Grid {}'.format(k-1), fontsize=75, y=0.92)
        
        for i in range(49):
            plt.subplot(7,7,i+1)
            plt.title('PSF № {}'.format(i), size=25)
            plt.imshow(hdul[k].data[i], norm=LogNorm(), cmap='PuRd')#norm=LogNorm(),
        if save:
            plt.savefig("PSF_GRID{}_Log_Norm.pdf".format(k-1))