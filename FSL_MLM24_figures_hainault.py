from pathlib import Path
from matplotlib import pyplot as plt

from icecream import ic # for debugging

# imoprt custom modules
import FSL_STM_UI as fsl_ui
import NavigatingTheMatrix as nvm




'''
This script is used to generate the figures for Taylor's presentation
at MLM24.
'''

cwd = Path.cwd()



'''
Hainault. Pre and post dose of Si(001) with AsH3
Dual bias from 2016-12-21
'''

Si_scans = {}
Si_scans_FSL = {}

# loop through the files in Si(001)-AsH3 examples folder

for i, file in enumerate(Path.joinpath(cwd, 'hainault_examples').iterdir()):
    if file.suffix != '.mtrx':
        print(file)
        
        if i == 0:
            # load matrix files
            Si_scans[f'hainault_{i}'] = nvm.STM( str(file) , None, None, None, standard_pix_ratio=512/100)
            print('cleaning up')
            Si_scans[f'hainault_{i}'].clean_up(Si_scans[f'hainault_{i}'].trace_up, 'trace up', plane_level=True)
            Si_scans[f'hainault_{i}'].clean_up(Si_scans[f'hainault_{i}'].retrace_up, 'retrace up', plane_level=True)
            print('correcting hysteresis')
            Si_scans[f'hainault_{i}'].trace_up_proc, Si_scans[f'hainault_{i}'].retrace_up_proc, corrected = Si_scans[f'hainault_{i}'].correct_hysteresis(Si_scans[f'hainault_{i}'].trace_up_proc, Si_scans[f'hainault_{i}'].retrace_up_proc, 'trace up')       
            fwd = Si_scans[f'hainault_{i}'].trace_up_proc
            bwd = Si_scans[f'hainault_{i}'].retrace_up_proc
            Si_scans_FSL[f'hainault_{i}'] = fsl_ui.FSL_Scan(str(file)[100:-6],'Si', fwd, bwd, size=200)
         
        #if i == 1:
            # load matrix files
        #    Si_scans[f'hainault_{i}'] = nvm.STM( str(file) , None, None, None, standard_pix_ratio=512/100)
            # plane level and scan line align and correct hysterisis
        #    print('cleaning up')
        #    Si_scans[f'hainault_{i}'].clean_up(Si_scans[f'hainault_{i}'].trace_down, 'trace down', plane_level=True)
        #    Si_scans[f'hainault_{i}'].clean_up(Si_scans[f'hainault_{i}'].retrace_down, 'retrace down', plane_level=True)
        #    print('correcting hysteresis')
        #    Si_scans[f'hainault_{i}'].trace_down_proc, Si_scans[f'hainault_{i}'].retrace_down_proc, corrected = Si_scans[f'hainault_{i}'].correct_hysteresis(Si_scans[f'hainault_{i}'].trace_down_proc, Si_scans[f'hainault_{i}'].retrace_down_proc, 'trace down')
        #    fwd = Si_scans[f'hainault_{i}'].trace_down_proc
        #    bwd = Si_scans[f'hainault_{i}'].retrace_down_proc
        #    Si_scans_FSL[f'hainault_{i}'] = fsl_ui.FSL_Scan(str(file)[100:-6],'Si', fwd, bwd, size=200)

for scan in Si_scans_FSL.values():
    plt.imshow(scan.scan_fwd, cmap='afmhot')
    plt.show()
    # prompt user to type in number of classes
    n_classes = int(input('How many classes are there?'))
    # create predictor object
    pred = fsl_ui.Predictor(scan, num_classes=n_classes, num_labels=1)
    pred.label_support_set(transparency=0.5)
    pred.label_anomalies()
    pred.predict()
    print(pred.defect_numbers()) # print number of defects in each class
    pred.display_image_with_mask('full', display_bwds = True, alpha=0.55, colormap='Set1')
    # save defect mask
    #np.save(f'{pred.scan.name}_mask.npy', pred.defect_mask)
    # label support set
    pred.save_coords_to_csv()

'''
example_scan = FSL_Scan('default_2020Mar05-185936_STM-STM_Spectroscopy--29_4_','Ge', example_array_fwd, example_array_bwd, size=50) # Ge(001)
   
example_pred = Predictor(example_scan, num_classes=7, num_labels=1)
example_pred.label_support_set(transparency=0.5)
example_pred.label_anomalies()
example_pred.predict()
print(example_pred.defect_numbers()) # print number of defects in each class
#    example_pred.display_image_with_mask('full', display_bwds = False, alpha=0.4)
example_pred.display_image_with_mask('full', display_bwds = True, alpha=0.4, colormap='Set1')
example_pred.save_coords_to_csv()
'''