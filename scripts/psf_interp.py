import datetime
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
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import cv2
import astropy
from astropy.io import fits
from astropy.table import Table




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
        
        return np.pad(image, ((0, n-y_size%n),(resid, n-y_size%n)), 'constant',constant_values=((0, 0),(0, 0)))
    else:
        return np.pad(image, ((resid, x_size%n),(0, x_size%n)), 'constant',constant_values=((0, 0),(0, 0)))
    
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

    X = np.linspace(x_min, x_max)

    Y = np.linspace(y_min, y_max)

    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation

    interp = LinearNDInterpolator(psfs_centres, z)
    print(list(zip(x, y)))
    print(z)

    Z = interp(X, Y)
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(20,20))
    
    plt.pcolormesh(X, Y, Z, shading='auto')

    plt.plot([i[0] for i in psfs_centres], [i[1] for i in psfs_centres], "ok", label="input point")

    plt.legend()

    plt.colorbar()

    plt.axis("equal")
    plt.savefig('linear_interp.png')
    plt.show()
    
    print(Z)
    return Z

def crop_image_and_blur(image, n, psf_size, hdul):
    
    """
    image: np.array
    
    n: int
    
    Size of psfs grid
    (For example: if grid 3x3, n == 3)
    
    psf_size: int
    
    How much pixels take from the center psf 
    (For example: if psf_size = 2, psf is 4x4)
    
    hdul: hdul of fits file
    
    """
    
    psf_size=5#5
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
            
def plot_psfs(hdul, j, save = True):
    
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
    
    plt.figure(figsize=(55,55))
    plt.rcParams.update({'font.size': 25})
    plt.suptitle('PSF Grid {}'.format(j-1), fontsize=75, y=0.92)


    for i in range(49):
        plt.subplot(7,7,i+1)
        plt.title('PSF № {}'.format(i), size=25)
        plt.imshow(hdul[j].data[i], norm=LogNorm(), cmap='PuRd')#norm=LogNorm(),
    if save:
        plt.savefig("PSF_GRID{}_Log_Norm.pdf".format(j-1))