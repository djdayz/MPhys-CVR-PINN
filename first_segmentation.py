from nipype.interfaces import fsl
import os

t1path = '/Users/mac/PycharmProjects/pythonMPhysproject/N4corrected_T1w/T1w_N4_shr3.nii'
outdir = '/Users/mac/PycharmProjects/pythonMPhysproject/first'
os.makedirs(outdir, exist_ok=True)

os.chdir(outdir)

first = fsl.FIRST(
    in_file=t1path,
    out_file= "firstsegmented",
    brain_extracted=True,
    output_type='NIFTI'
)
result = first.run()

print(f"Combined 4d segmentation: {result.outputs.segmentation_file}")
print(f"Label map: {result.outputs.original_segmentations}")
print(f"vtk surfaces: {result.outputs.vtk_surfacessf}")

