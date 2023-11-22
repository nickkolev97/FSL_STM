import numpy as np
import torch
import torch.nn.functional as F
import patchify as pat
import cv2
from matplotlib import pyplot as plt
import FSL_model as mo
from pathlib import Path

# debugging module
from icecream import ic

'''
Contains the user interface for the the FSL.
Designed to run on CPU so needs asjustments to use GPU.
However, for most day to day uses, CPU speeds should be fine.
'''

class FSL_Scan(object):
    '''
    This class takes in a numpy array of an STM scan and with some 
    meta info such as scan size.
    Also performs some clean up procedure on the scans if wanted 
    (these are mostly if it came from an unprocessed .png).

    Parameters:
    - name: user chooses name for this scan.
    - surface: the type of surface this is. One of 'TiO2', 'Si(001)', 'Ge(001)'.
    - scan_fwd: numpy array of forward scan
    - scan_bwd: if the backward scan is available, input it here, if not, leave this empty.
    - size: only needed if it's a .png file. Real width of scan in nm. 
            Assumes scan is square.

    Attributes:
    - self.name: name of scan
    - self.scan : numpy array of the scan.
    - self.size: real width of scan in nm. Assumes scan is square.
    - self.scan_fwd_repeated: True/False depending on if we just repeat
                              scan_fwd for scan_bwd. 

    Methods:
    - plane_level: plane levels the scans.
    - scan_line_algin: scan line aligns the scans.
    - hyst_correct: hysterisis correction on the forwards and backwards scans 
                    if both are present.
    -self.resample: resamples scan so that it has the same pixel to nm ratio as
                    the training data.

    Usage Example:
    ```python
    
    ```

    '''

    def __init__(self, name, surface, scan_fwd, scan_bwd=None, size=None):
        self.name = name
        self.surface = surface
        self.scan_fwd = scan_fwd
        if scan_bwd == None:
            self.scan_bwd = scan_fwd
            self.scan_fwd_repeated = True
        else:
            self.scan_bwd = scan_bwd
            self.scan_fwd_repeated = False
        self.size = size
        self.res0, self.res1 = scan_fwd.shape
        self._resample()

    def _resample(self):
        """
        If array is not of size (2^n, 2^n), n>5, then we want to resample it so that it is.
        This is just due to the way the segmentation algorithm works.
        Args:
            self

        Returns:
        """
        # find new resolution. This depends on what surface it is.
        if self.surface == 'Si':
            # 100nm to 512pixels
            
        
        new_res = resolutions[np.argmin(np.array(resolutions)-self.res0)]
        # turn numpy array to torch tensor for resampling
        fwd_tens = torch.tensor(self.scan_fwd)
       # fwd_tens = F.interpolate(fwd_tens, size=(new_res, new_res), mode='bilinear', align_corners=False)
        fwd_tens = F.interpolate(fwd_tens.unsqueeze(0).unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        fwd_array = fwd_tens.detach().numpy()
        if self.scan_fwd_repeated==True:
            bwd_array = fwd_array.copy()
        else:
            bwd_tens = torch.tensor(self.scan_bwd)
            bwd_tens = F.interpolate(bwd_tens, size=(new_res, new_res), mode='bilinear', align_corners=False)
            bwd_array = bwd_tens.detach().numpy()
        
        self.scan_fwd = fwd_array
        self.scan_bwd = bwd_array

        return
        
    def plane_level(self):
        """
        Performs a plane level on both forwards and backwards scans.

        Args:
            self

        Returns:
            None
        """
        leveled_scan_fwd = self._plane_level(self.scan_fwd)
        self.scan_fwd = leveled_scan_fwd

        if self.scan_fwd_repeated:
            # if we are using the fwd scan as the bwd too, then just use the array we just levelled.
            self.scan_bwd = leveled_scan_fwd
        else:
            # else, repeat the same procedure with the bwd scan.
            leveled_scan_bwd = self._plane_level(self.scan_bwd)
            self.scan_bwd = leveled_scan_bwd
     
        return

    def hyst_correct(self):
        """
        Performs a hysterisis correction on the forwards and backwards scans.

        Args:
            self

        Returns:
            None
        """

    def scan_line_align(self):
        """
        Performs a scan line align on both forwards and backwards scans.

        Args:
            self
        
        Returns:
            None
        """

    def _plane_level(self, array):
        """
        Performs plane level on whatever array is given.
        Args:
            array: array to plane level
        Returns:
            array_leveled: the plane leveled array
        """
        # simple plane level. Assumes the whole scan is on same plane.
        res = array.shape[0]
       # print('array shape', array.shape)
        # plane level using these arrays/masks
        a = np.ogrid[0:res,0:res]
        x_pts = np.tile(a[0],res).flatten()
        y_pts = np.tile(a[1],res).flatten()
        z_pts = array.flatten()
        
        X_data = np.hstack( ( np.expand_dims(x_pts, axis=1) , np.expand_dims(y_pts,axis=1) ) )
        X_data = np.hstack( ( np.ones((x_pts.size,1)) , X_data ))
        Y_data = np.reshape(z_pts, (x_pts.size, 1))
        fit = np.dot(np.dot( np.linalg.pinv(np.dot(X_data.T, X_data)), X_data.T), Y_data)
        
        # print("coefficients of equation of plane, (a1, a2) 2: ", fit[0], fit[1])
        # print("value of intercept, c2:", fit[2] )
              
        # make a grid to use for plane subtraction (using numpy's vectorisation)
        x = np.linspace(0,res, num=res, endpoint = False, dtype=int)
        y = np.linspace(0,res, num=res, endpoint = False, dtype=int)
        grid = np.meshgrid(x,y)
        
        # perform plane subtraction
        array_leveled = array - fit[2]*grid[0] - fit[1]*grid[1] - fit[0]
      
        '''
        # this is code for doing a second degree surface subtraction. Stick with first order for now
        # because the variance when using second order is larger (unsurprisingly, this is what we expect)
        
        x_ptsy_pts, x_ptsx_pts, y_ptsy_pts = x_pts*y_pts, x_pts*x_pts, y_pts*y_pts

        X_data = np.array([x_pts, y_pts, x_ptsy_pts, x_ptsx_pts, y_ptsy_pts]).T  # X_data shape: n, 5
        Y_data = z_pts

        reg = linear_model.LinearRegression().fit(X_data, Y_data)

        print("coefficients of equation of plane, (a1, a2, a3, a4, a5): ", reg.coef_)

        print("value of intercept, c:", reg.intercept_)
        
        array_leveled2 = array - reg.coef_[0]*grid[0] -  reg.coef_[1]*grid[1] - reg.coef_[2]*grid[0]*grid[1] - reg.coef_[3]*grid[0]*grid[0]- reg.coef_[3]*grid[1]*grid[1] - reg.intercept_
        plt.imshow(array_leveled2)
        plt.show()
        
        print(np.mean(array_leveled2), np.var(array_leveled2[labels==largest_area_label]) , np.mean(array_leveled), np.var(array_leveled[labels==largest_area_label]) )
        '''
     
        return array_leveled


class Predictor(object):
    '''
    Takes an STM scans and outputs a segmented image. To do this, it prompts the user for some labels
    on the scan.

    Parameters:
    - scan: FSL_scan object from this file. Contains data about the scan such as real size, 
            and the scan itself as a numpy array. 
    - num_classes: number of classes of defects in this scan.
    - num_labels: number of labels you want to give per class for the FSL network.
    - max_area: maximum area (in pixels**2) of a defect to classify

    Attributes:
    - self.scan : numpy array of the scan.
    - self.num_classes:  number of classes of defects in this scan.
    - self.num_labels: number of labels you want to give per class for the FSL network.
    - self.defect_mask: a binary map of where the defects in the scan are.
    - self.defect_coords: dictionary of all defect coords except the ones that are too large.
                         Keys are integer labels (that come from the connected components 
                         analysis) and values are their pixel coords.
    - self.segmented_map: a final segmented map of the different features. Numpy array of shape 
                          (num_classes+3, res, res). +3 for anomalies, large defects, and lattice.
    - self.support_set: a dictionary with the support set. It's of the form:
                        self.support_set = {'image': numpy array of shape (number of crops, num_channels, res, res), 
                                            'target': numpy array of shape (number of crops, y_values for this episode),
                                            'coords': numpy array of shape (number of crops, 2),
                                            'idxs': numpy array of idx of each support crop, idx being from the 
                                                connected component analysis. shape = (num crops, 1) }
    - self.query_set: a dictionary with the query set. It's of the form:
                      self.query_set = {idx: {'target': n, 'image': torch.tensor, 'coord': (x,y), 'idx':idx} }
    - self.anom_set: dictionary of anomalies. Keys are integer labels (that come from the connected components
                     analysis) and values are info about defect such as pixel coordinates. It's of the form:
                     anom_set = {defect_idx: {'target': label, 'image': None, 'coord': self.defect_coords[defect_idx], 'idx': defect_idx}}
    - self.large_defect_coords: dictionary of defects that are too large and their coords. Keys are integer 
                                labels (that come from the connected components analysis) and values are 
                                their pixel coords.Default for this is 20 pixels**2.
    - self.UNet: the UNet used to find defects. This depends on the surface type.
    - self.win_size: to improve accuracy of UNet we use windowing. This is where we split the scan into overlapping
                     patches and get a prediction from each patch then combine them for final. win_size is the size
                     of the patches used.
    - self.fsl_model: the few-shot learning model used to classify defects.
    - self.fsl_lightning_model: the fsl model wrapped in a pytorch lightning model.
    TODO: SHOULD CHANGE THIS SO WE JUST HAVE AN FSL MODEL, NO NEED FOR BOTH (BUT NEED RETRAINING OF THE MODEL FOR THAT AND SAVING THE RIGHT WEIGHTS)
    - self.labels: the labels from the connected component analysis on the output of the UNet.

    Methods:
    - _get_defect_mask: finds defects in the lattice and returns as binary map.
    - label: opens the scan in openCV with the binary map overlaid and allows user to click to label
             the query set.
    - maxminnorm: max/min normalises the input.
    - mean0norm: makes mean of every channel 0.
    - _click_event: function called by openCV to save mouse click for the label method.
    - segment_map: turns predictions into a one-hot encoded map.
    - _load_model: loads weights onto pytorch model.
    - display_image_with_mask: displays the fwd scan with either the defect mask or fully segmented map overlaid.
    - save_coords_to_csv: saves the coordinates and their label to a .csv file.

    Usage Example:
    
    '''  

    def __init__(self, scan, num_classes, num_labels, max_area = 20):
        self.scan = scan
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.max_area = max_area
        ic(self.scan.scan_fwd.shape)
        self.UNet = mo.UNet()
        cwd = Path.cwd()
        if scan.surface == 'Si':
            UNet_file_path = Path.joinpath(cwd,'models', 'UNet_Si.pth')
            self.UNet = self._load_model(self.UNet, UNet_file_path)
            self.win_size = 32
        elif scan.surface == 'Ge':
            UNet_file_path = Path.joinpath(cwd,'models', 'UNet_Ge.pth')
            self.UNet = self._load_model(self.UNet, UNet_file_path)
            self.win_size = 32
        elif scan.surface == 'TiO2':
            UNet_file_path = Path.joinpath(cwd,'models', 'UNet_TiO2.pth')
            self.UNet = self._load_model(self.UNet, UNet_file_path)
            self.win_size = 128
        
        self.UNet.eval()

        fsl_file_path = Path.joinpath(cwd,'models', 'FSL_protonet_(3,15)_40pix_SiGe_inv.pth')        
        self.fsl_model = mo.PrototypicalNetwork()
        self.fsl_lightning_model = mo.FewShotLearner(self.fsl_model, n_way=num_classes)
        self.fsl_lightning_model = self._load_model(self.fsl_lightning_model, fsl_file_path)
        self.fsl_lightning_model.eval()

        self.defect_mask, self.defect_coords, self.large_defect_coords = self._get_defect_mask()
        res0, res1 = self.scan.scan_fwd.shape
        self.fully_segmented_map = np.zeros((num_classes+2,res0,res1))
        
        self.support_set = {}
        self.query_set = {}
        self.anom_set = {}

    def _load_model(self, model, file_path):
        """
        Loads weights onto torch model.
        Args: 
            model: model to load weights onto
            file_path: location of weights (as a .pth file)
        Returns:
            model: model with desired weights.
        """
        model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu') ) )
        return model

    def maxminnorm(self, array):
        """
        Does a max/min normalisation on input tensor.

        Args:
            self:
            array: array with crops to normalise. Should be of shape (n,1,win_size,win_size), 
                   where n is the number of crops to normalise.

        Return:
            array: normalised array of shape (n,1,win_size,win_wize)
        """
        min_filled = torch.min(torch.min(array[:,0,:,:], dim=1)[0],dim=1)[0].unsqueeze(-1).unsqueeze(-1)
        max_filled = torch.max(torch.max(array[:,0,:,:], dim=1)[0],dim=1)[0].unsqueeze(-1).unsqueeze(-1)

        array = (array[:,0,:,:]-min_filled)/(max_filled-min_filled)

        return array.unsqueeze(1)

    def mean0norm(self, array):
        """
        Makes mean of every channel of array equal to 0.
        Args:
            array: numpy array to be normalised.

        Returns:
            array: normalised array.
        """
        # find maximums and minimums
        mean0 = np.mean(array[0,:,:])
        mean1 = np.mean(array[0,:,:])
        
        array[0,:,:] = array[0,:,:]-mean0
        array[1,:,:] = array[1,:,:]-mean1
        
        return array

    def _get_defect_mask(self):
        """
        Finds the defects on the surface
        
        Args:
            self
        
        Returns:
            defect_mask: a segmented mask of where the defects in the scan are.
            defect_coords: the pixel coordinates of the defects.
        """
        
        array = self.scan.scan_fwd.copy()
        res = array.shape[0]
        win_size2 = self.win_size//2
        sqrt_num_patches = ((res-self.win_size)//(self.win_size//2)+1)

        # cut up the array into patches of size win_size
        patches = np.reshape( pat.patchify(array, (self.win_size, self.win_size), step = self.win_size//2), ( ( sqrt_num_patches**2 , 1, self.win_size,self.win_size) ) )
        # normalise and turn to torch tensor
        patches = self.maxminnorm(torch.tensor(patches).float())

        # find defects
        torch.manual_seed(0) # need to seed or conv layers aren't deterministic
        unet_prediction = self.UNet(patches)
        unet_prediction = torch.reshape(unet_prediction, (sqrt_num_patches, sqrt_num_patches, 2, 1, self.win_size, self.win_size))
        prediction = torch.zeros((2,res,res))
        # To get rid of edge effects of the U-Net, we take smaller steps so each crop overlaps and then take an average over the crops
        # takes a bit longer to compute but is more accurate
        for i in range(sqrt_num_patches):
            for j in range(sqrt_num_patches):
                prediction[:,i*win_size2:(i*win_size2)+self.win_size, j*win_size2:(j*win_size2)+self.win_size] = prediction[:,i*win_size2:(i*win_size2)+self.win_size, j*win_size2:(j*win_size2)+self.win_size] + unet_prediction[i,j,:,0,:,:]     

        defect_mask = torch.argmax(prediction,dim=0)
        
        # get pixel coordinates of all defects
        connected_comps = cv2.connectedComponentsWithStats(defect_mask.detach().numpy().astype(np.uint8))
        (numLabels, self.labels, stats, centroids) = connected_comps
        defect_coords = {}
        large_defect_coords = {}
        # loop over the number of unique connected component labels
        for i in range(0, numLabels):
            if i == 0:
                pass # first one is background so ignore
            # otherwise, we are examining an actual defect
            else:
                area = stats[i, cv2.CC_STAT_AREA]
                if area<self.max_area:
                    defect_coords[i] = centroids[i]
                    # centroids in form np.array([x,y]) (x=col idx, y=row idx)
                else:
                    large_defect_coords[i] = centroids[i]

    
        
        '''    
        plt.imshow(defect_mask.detach().numpy())
        plt.show()
        
        plt.imshow(pred2.detach().numpy())
        plt.show()
        '''

        return defect_mask, defect_coords, large_defect_coords

    def _label(self, transparency = 0.5):
        """
        Opens up an openCV window with the segmented mask of the defects overlaid.
        Ask user to give labels either for support or query set.

        Args:
            self
            transparency: how transparent the overlaid segmentation mask is, between 0 and 1.
                          Cannot be 0.5.

        Returns:
            list_of_coords: the list of coordinates selected by user

        """

        # NOTE/ TODO: I think it'd be better if there were squares around features rather than the
        #             defect mask overlaid. Should change this later.

        if transparency==0.5:
            raise Exception('transparency cannot be equal to 0.5')

        # display scan with mask overlaid
        plt.imshow(self.scan.scan_fwd, cmap='afmhot')
        plt.imshow(self.defect_mask, alpha=transparency, interpolation='none', cmap='tab20')
        # save figure
        plt.savefig('outputs\{}_defect_mask.png'.format(self.scan.name), transparent=True, pad_inches=0)
        plt.clf() #clear current figure
        # load in CV
        image = cv2.imread('outputs\{}_defect_mask.png'.format(self.scan.name))
        
        imax = np.max(image)
        imin = np.min(image)
        if imax!=255 or imin!=0:
            image = 255*(image-imin)/(imax-imin)
            
        # Create a named window to display the canvas
        cv2.namedWindow('Labeller')
        
        # list of coords of defects
        list_of_coords = []
        # Set the callback function for mouse events
        cv2.setMouseCallback('Labeller', self._click_event, param = [list_of_coords] )

        while True:
            cv2.imshow('Labeller', image)
            key = cv2.waitKey(1)
            if key == 27:  # Press Esc to exit
                break

        cv2.destroyAllWindows()

        return list_of_coords
    
    def label_support_set(self, transparency = 0.7):
        """
        Opens up an openCV window with the segmented mask of the defects overlaid.
        Ask user to give labels for the query set.
        Labelling should be done as follows:
            - The number of classes in scan is given at beginning (N).
            - The number of labels per class is also decided at beginning (K).
            - The first K labels given here (mouse clicks) correspond to the 
              defects in class 0, the second K labels given correspond to the 
              defect in class 1, etc.

        Args:
            self
            transparency: how transparent the overlaid segmentation mask is, between 0 and 1.
                          Cannot be 0.5.

        Returns:
            support_set: dictionary with the support set. It's of the form:        
                        self.support_set = {'image': numpy array of shape (number of crops, num_channels, res, res), 
                                            'target': numpy array of shape (number of crops, y_values for this episode),
                                            'coords': numpy array of shape (number of crops, 2),
                                            'idx': numpy array of idx of each support crop, idx being from the 
                                                connected component analysis. shape = (num crops, 1) }
            query_set: dictionary with the query set. It's of the form:
                       self.query_set = {idx: {'label': n, 'image': torch.tensor, 'coord': (x,y), 'idx':idx} }
        """
        list_of_coords = self._label(transparency)

        # define support set coords list
        support_set_coords = []

        # prepare the array to crop the defect out from
        # since the defect may be close to the border we want to add
        # a small padding of about 10% of the image res.
        pad_size = int(round(self.scan.scan_fwd.shape[0]/10,0))
        scan_fwd = np.pad(self.scan.scan_fwd, pad_width = pad_size, mode = 'median')
        scan_bwd = np.pad(self.scan.scan_bwd, pad_width = pad_size, mode = 'median')
        scan = np.stack((scan_fwd, scan_bwd), axis=2)
        # now we need to find which label corresponds to which defect.
        # Find closest defect to each label and save it to the dictionary.
        defect_coords = list(self.defect_coords.values())
        defect_coords_array = np.array(list(self.defect_coords.values()))
        defect_coord_idx = list(self.defect_coords.keys())
        
        for coord in list_of_coords:
            dists = np.sqrt( np.sum((coord-defect_coords_array)**2, axis = 1) )
            # find the index of the nearest defect to the label. This index is w.r.t
            # the 0th axis of the defect_coords_array 
            min_dist_idx = np.argmin(dists) 
            # now we want the idx of this w.r.t the indices given to the defects
            # by the connect component analysis since this is how we refer to them
            # within the dictionaries.
            defect_idx = defect_coord_idx[min_dist_idx]
            # now we want to get what label this is (from 0 to num_classes-1). This is
            # figured out by finding which number label (target) it is.
            label = list_of_coords.index(coord)//self.num_labels
            image = self._get_image(self.defect_coords[defect_idx], scan, pad_size)
            self.support_set[defect_idx] = {'target': label, 'image': image, 'coord': self.defect_coords[defect_idx], 'idx': defect_idx}
            support_set_coords.append(self.defect_coords[defect_idx])

        # update the structure of the dictionary so it can be used for predictions.
        self.support_set = self._update_structure(self.support_set)

        # Now find the query set dict which is just the leftover defects.
        # Use set operations for this.
        # Convert the arrays to tuples for set operations.
        defect_coords_set = {tuple(coord) for coord in defect_coords}
        support_set_coords_set = {tuple(coord) for coord in support_set_coords}

        # Get the remaining coordinates
        query_coords_set = defect_coords_set-support_set_coords_set

        # Convert the set to a list
        query_set_coords_tuples = list(query_coords_set)
        query_set_coords = [np.array(coord) for coord in query_set_coords_tuples]

        # get list of indices from connected comp analysis
        indices = list(self.defect_coords.keys())
        for coord in query_set_coords:
            # get its idx from the connected component analysis
            idx = indices[np.where(np.sum(defect_coords_array==coord,axis=1)==2)[0][0]]
            image = self._get_image(coord, scan, pad_size).unsqueeze(0)
            # add to query set. 'target' is None still as it hasn't been predicted
            self.query_set[idx] = {'target': None, 'image': image, 'coord': coord, 'idx': idx}

        return
    
    def label_anomalies(self, transparency=0.7):
        """
        Opens up an openCV window with the segmented mask of the defects overlaid.
        Allows user to select anomalies they want to disregard.
        This is useful if there's only one or two of them so not enough to make up a class of
        their own.

        Args:
            self
        
        Returns:
            anomaly_coords: coords of the defects to ignore.
        """
        # label coordinates of anomalies using openCV
        list_of_coords = self._label(transparency)

        # define anomalies set coords list
       # anom_set_coords = []

        # now we need to find which label corresponds to which defect.
        # Find closest defect to each label and save it to the dictionary.
        defect_coords = list(self.defect_coords.values())
        defect_coords_array = np.array(list(self.defect_coords.values()))
        defect_coord_idx = list(self.defect_coords.keys())
        
        for coord in list_of_coords:
            dists = np.sqrt( np.sum((coord-defect_coords_array)**2, axis = 1) )
            # find the index of the nearest defect to the label. This index is w.r.t
            # the 0th axis of the defect_coords_array 
            min_dist_idx = np.argmin(dists) 
            # now we want the idx of this w.r.t the indices given to the defects
            # by the connect component analysis since this is how we refer to them
            # within the dictionaries.
            defect_idx = defect_coord_idx[min_dist_idx]
            # since these are all anomalies, their label is num_classes
            label = self.num_classes
            # don't need image as it won't be getting classified by FSL.
            self.anom_set[defect_idx] = {'target': label, 'image': None, 'coord': self.defect_coords[defect_idx], 'idx': defect_idx}
            

        return None
    
    def _click_event(self, event, x, y, flags, param):
        """
        Registers where the mouse click happened and saves the pixel coordinate to an array
        Labels are made with left mouse click.
        NOTE: The previous label can be deleted with a mouse wheel click.
        Press 'esc' to exit once done.
        
        Args:
            event: mouse click
            x, y : pixel coordinateso of mouse click
            param: 0th is the list of coords to be filled up.
        
        Return:
            coord_array: array of coordinates that were clicked on.
        """

        list_of_coords = param[0]

        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
    
            # displaying the coordinates
            # on the Shell
            ic(x, y)
            list_of_coords.append( [x,y] )

            # displaying the coordinates
            # on the image window
           # font = cv2.FONT_HERSHEY_SIMPLEX
           # cv2.putText(img, '.', (x-5,y), font,
           #             1, (255, 0, 0), 2)
           # cv2.imshow('image', img)

        # checking for right mouse clicks
        # this isn't used right now, but I've left it in in case there's a use for it later.
        # Could be used to label 2 different features simultaneously, I guess.
        #if event==cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
        #    print(x, ' ', y)

        if event==cv2.EVENT_MBUTTONDOWN:
            # use mousewheel to delete the previous click if it was wrong.
            previous_coord = list_of_coords[-1]
            list_of_coords.remove(previous_coord)
            print( previous_coord, ' deleted' )

    def _get_image(self, coord, scan, pad_width):
        """
        Returns a crop of the defect as a torch tensor
        with mean of 0.

        Args:
            self
            coord: pixel coordinates of the defect.
            scan: numpy array of the scan with 2 channels
            pad_width: width of padding added onto scan

        Returns:
            crop: crop of the defect with 2 channels.
        """    

        # TODO: crop size should change depending on size of defect (with some minimum and maximum probably),
        # rather than due to surface, with some possible input from user? For now, just use automatic based on surface.
        if self.scan.surface == 'Si':
            crop_size = 11
        if self.scan.surface == 'Ge':
            crop_size = 11
        if self.scan.surface == 'TiO2':
            crop_size = 58
        hcs = int(round(crop_size/2,0)) # half crop size
        x, y = coord.astype(np.uint16) + pad_width #should this be swapped?
        crop = np.transpose(scan[y-hcs:y+hcs-1, x-hcs:x+hcs-1,:], (2,0,1)).copy()
        # normalise (mean to 0)
        crop = self.mean0norm(crop)
        # turn to torch.tensor
        crop = torch.tensor(crop).float() 

        return crop

    def predict(self):
        """
        Uses the query set to classify the different defects.

        Args:
            self

        Returns:
            segmented_image: final segmented image with key
            class_coords: dictionary. Keys are the classes (as integers) with 
                          coordinates of the defects as the values.

                          
        """

        for defect in self.query_set.values():
            defect['target'] = int(torch.argmax(self.fsl_model(defect, self.support_set)))

        self.segment_map()

        return

    def segment_map(self):
        """
        Turns the predictions in self.query_set, and self.support_set into a fully segmented map.

        Args:
            self
        
        Returns
        """
        for defect in self.query_set.values():
            self.fully_segmented_map[defect['target'],:,:] += (self.labels==defect['idx'])
        for j, idx in enumerate(self.support_set['idxs']):
            self.fully_segmented_map[self.support_set['target'][j],:,:] += (self.labels==idx)
        for defect in self.anom_set.values():
            self.fully_segmented_map[defect['target'],:,:] +=(self.labels==defect['idx'])

        # define colour for lattice background
        self.fully_segmented_map[-1,:,:] += (self.labels==0)
        '''
        TODO: ADD DISPLAY FOR ANOMALIES AND LARGE DEFECTS (channels -2 and -3)
        '''
      
        return

    def _update_structure(self, d):
        """
        Turns self.support_set dictionary from 
        self.support_set = {integer: {'target': n, 'image': numpy array, 'coord': (x,y), 'idx': idx} }
        structure to 
        self.support_set = {'image': numpy array of shape (number of crops, num_channels, res, res), 
                            'target': numpy array of shape (number of crops, y_values for this episode),
                            'coords': numpy array of shape (number of crops, 2),
                            'idxs': numpy array of idx of each support crop, idx being from the 
                                   connected component analysis. shape = (num crops, 1)
                            'classlist': list of integers from 0 to num_classes-1. This is needed due to structure
                                         of the fsl_networks and their datasets during training.
        Args:
            self
            d: dictionary to collate
        Return:
            new_d: dictionary with updated structure.
        """
        list_of_dicts = list(d.values())
        images = []
        targets = []
        coords = []
        idxs = []
        for item in list_of_dicts:
            images.append(item['image'])
            targets.append(item['target'])
            coords.append(item['coord'])
            idxs.append(item['idx'])
                
        images = torch.stack(images, dim=0)
        targets = np.stack(targets, axis=0)
        coords = np.stack(coords, axis=0)                       
        idxs = np.stack(idxs)
          
        return {'image': images, 'target':targets, 'coords': coords, 'idxs': idxs, 'classlist': [i for i in range(self.num_classes)]}

    def display_image_with_mask(self, mask, alpha=0.5):
        """
        Displays the forward scan with either the defect or fully segmented mask.
        Args:
            self:
            mask: 'defect' or 'full' tell you whether to use the fully segmented
                  or defect mask.
            alpha: controls transparency of mask.
        Returns:
            None
        """
        if mask == 'defect':
            mask_ = np.argmax(self.defect_mask, axis=0)
        elif mask == 'full':
            mask_ = np.argmax(self.fully_segmented_map, axis = 0)
        else:
            raise Exception('mask must be one of "defect" or "full"')

        # display scan with mask overlaid
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.scan.scan_fwd, cmap='afmhot')
        ax[0].imshow(mask_, alpha=alpha, interpolation='none', cmap='tab20')
        ax[1].imshow(self.scan.scan_fwd, cmap='afmhot')
        plt.show()

        return 

    def save_coords_to_csv(self, filepath=None):
        """
        Saves the coordinates and their label to a .csv file.
        Args:
            self:
            filepath: can specify filepath if wanted. Otherwise, it'll save it in this folder and as
                      "scan_name_coords.csv".
        Returns:
            None
        """
        # make an array of coordinates. Array should be of the form:
        # [[idx],[x],[y],[n]] where idx is its index from connected component
        # analysis, x and y its coordinates, and n its label.
        # first do support set
        coords_array = self.support_set['coords']
        labels_array = np.expand_dims(self.support_set['target'],axis=-1)
        idxs_array = np.expand_dims(self.support_set['idxs'],axis=-1)
        csv_array = np.concatenate((idxs_array, coords_array, labels_array), axis = 1)
        # now do query set
        for defect in self.query_set.values():
            coords = np.expand_dims(defect['coord'],axis=0)
            label = np.array( [[defect['target']]])
            idx = np.array([[defect['idx']]])
            array = np.concatenate((idx, coords, label), axis = 1)
            csv_array = np.concatenate((csv_array,array), axis=0)
        
        np.savetxt("outputs\{}_coords_and_labels.csv".format(self.scan.name), csv_array, delimiter=",")

        return


if __name__ == "__main__":
    cwd = Path.cwd()
   # file_path = Path.joinpath(cwd, 'example_arrays', 'm70.npy')
   # file_path = Path.joinpath(cwd, 'example_arrays', 'm235.npy')
    file_path = Path.joinpath(cwd, 'example_arrays', 'm63_ori_1.png')
   # example_array = np.load(file_path)
    example_array = plt.imread(file_path)
    example_scan = FSL_Scan('m63','TiO2', example_array, size=20)
    example_pred = Predictor(example_scan, num_classes=2, num_labels=1, max_area=70000)
    example_pred.label_support_set()
    example_pred.predict()
    example_pred.display_image_with_mask('full', alpha=0.6)
    example_pred.save_coords_to_csv()
