import os
import SimpleITK as sitk


def save_nifti(arrays, image_org, filename):
    org_path = os.path.join("/home/dongkyu/Datasets/AAPM_3D/" + image_org + "/full_3mm", image_org + "_FD_3mm.nii.gz")
    image_org = sitk.ReadImage(org_path)
    images = sitk.GetImageFromArray(arrays)
    images.CopyInformation(image_org)
    sitk.WriteImage(images, filename)