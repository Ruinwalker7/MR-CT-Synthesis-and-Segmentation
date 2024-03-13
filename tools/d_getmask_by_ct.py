import argparse
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
from operator import mul

# def remove_bed(ct):
#     pass

# def segment(input_mr,input_ct, output_mask=None):
#     mr = sitk.ReadImage(input_mr)
#     ct = sitk.ReadImage(input_ct)
#     ct = remove_bed(ct)#CT除床
#     image = sitk.InvertIntensity(sitk.Cast(ct,sitk.sitkFloat32))
#     mask = sitk.OtsuThreshold(image)
#     dil_mask = sitk.BinaryDilate(mask, (10, 10, 1))
#     component_image = sitk.ConnectedComponent(dil_mask)
#     sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
#     largest_component_binary_image = sorted_component_image == 1
#     mask_closed = sitk.BinaryMorphologicalClosing(largest_component_binary_image, (12, 12, 12))
#     dilated_mask = sitk.BinaryDilate(mask_closed, (10, 10, 0))
#     filled_mask = sitk.BinaryFillhole(dilated_mask)
#     mask = sitk.GetArrayFromImage(filled_mask)
#     mr_ar = sitk.GetArrayFromImage(mr)
#     mask[mr_ar==0] = 0
#     filled_mask = sitk.GetImageFromArray(mask)
#     sitk.WriteImage(filled_mask, output_mask)

# if __name__ == '__main__':
#     dataroot = 'D:\Research\dataset\XiangYa\suplement/resample_nii' #文件夹路径
#     outroot = 'D:\Research\dataset\XiangYa\suplement/resample_nii'
#     patients = os.listdir(dataroot)
#     # patients = ['0001510226']
#     for i in patients:
#         if not os.path.exists(outroot+'/'+i):
#             os.makedirs(outroot+'/'+i)
#         # 根据MR和CT生成MASK
#         segment(input_mr = dataroot + '/' + i + '/mr.nii.gz',input_ct = dataroot + '/' + i + '/ct.nii.gz', output_mask=outroot+'/'+i+'/mask1.nii.gz')
#         print(i+' OK')


import os
import SimpleITK as sitk
import numpy as np


def segment(input_image, output_mask=None, radius=(12, 12, 12), return_sitk=False):
    input_image = sitk.ReadImage(input_image)
    # 限制大小
    image = sitk.GetArrayFromImage(input_image)
    image[image >= 5000] = 5000
    image[image < 0] = 0
    image = sitk.GetImageFromArray(image)
    image.CopyInformation(input_image)
    image = sitk.InvertIntensity(sitk.Cast(image, sitk.sitkFloat32))

    # mask = sitk.YenThreshold(image)
    # mask = sitk.MaximumEntropyThreshold(image)

    mask = sitk.RenyiEntropyThreshold(image)
    eroded_mask = sitk.BinaryErode(mask, (4, 4, 0))
    # 二值膨胀操作sitk.BinaryDilate对掩膜进行膨胀，以填充掩膜中的空洞
    dil_mask = sitk.BinaryDilate(eroded_mask, (10, 10, 1))
    # 对膨胀后的掩膜进行连通组件分析，得到每个连通组件的标签
    component_image = sitk.ConnectedComponent(dil_mask)
    # 按对象大小排序的连通组件重标记，将最大的连通组件标记为1，其余的连通组件标记为其他
    sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1
    # 通过形态学闭运算sitk.BinaryMorphologicalClosing对最大连通组件进行闭运算，以去除一些小的孔洞
    mask_closed = sitk.BinaryMorphologicalClosing(largest_component_binary_image, radius)
    # 使用二值膨胀操作sitk.BinaryDilate对闭运算后的掩膜进行膨胀，以扩展掩膜的边界
    dilated_mask = sitk.BinaryDilate(mask_closed, (10, 10, 0))
    # 使用二值填充孔洞操作sitk.BinaryFillhole对膨胀后的掩膜进行填充，以填补一些孔洞
    filled_mask = sitk.BinaryFillhole(dilated_mask)
    # filled_mask = mask
    if return_sitk:
        return filled_mask
    else:
        sitk.WriteImage(filled_mask, output_mask)


# 根据MR获取MASK
# 利用CT 步骤：二值化，找最大联通区域，闭运算，膨胀，去掉mr的0值区域
# MaximumEntropyThreshold、RenyiEntropyThreshold、YenThreshold is good
if __name__ == '__main__':
    dataroot = 'D:\Research\dataset\XiangYa/resample_nii'  # 文件夹路径
    outroot = 'D:/Research/dataset/XiangYa/resample_nii'
    patients = os.listdir(dataroot)
    # patients = ['0001510226']
    for i in patients:
        if not os.path.exists(outroot + '/' + i):
            os.makedirs(outroot + '/' + i)
        # 根据MR生成MASK
        segment(input_image=dataroot + '/' + i + '/mr.nii.gz', output_mask=outroot + '/' + i + '/mask.nii.gz')
        # MR
        # resample(input = dataroot + '/' + i + '/mr.nii.gz', output=outroot+'/'+i+'/mr.nii.gz', spacing=(1,1,1))
        print(i + 'OK')
