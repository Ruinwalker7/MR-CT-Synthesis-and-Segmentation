import SimpleITK as sitk
import numpy as np
import itk
import os
from itkwidgets import compare, checkerboard

# 1. T2加权MR体积进行N4校正
def perform_n4_correction(input_path, output_path):
    image = sitk.ReadImage(input_path)

    # 创建N4BiasFieldCorrection对象
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    inputImage = sitk.ReadImage(input_path, sitk.sitkFloat32)
    image = inputImage
    corrected_image = corrector.Execute(image)

    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)

    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)

    sitk.WriteImage(corrected_image_full_resolution, output_path)


# 2. 将同一受试者的CT体积通过Elastix注册到T2加权MR体积
def perform_image_registration(t2_path, ct_path, output_path):
    print(ct_path)
    fixed_image = itk.imread(t2_path, itk.F)
    moving_image = itk.imread(ct_path, itk.F)
    print(fixed_image.GetOrigin())
    print(moving_image.GetOrigin())
    origin = fixed_image.GetOrigin()

    moving_image.SetOrigin(origin)
    parameter_object = itk.ParameterObject.New()
    parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(parameter_map_rigid)
    parameter_map_affine = parameter_object.GetDefaultParameterMap('affine')
    parameter_object.AddParameterMap(parameter_map_affine)
    # parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')
    # parameter_object.AddParameterMap(parameter_map_bspline)
    # Call registration function
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        log_to_console=False)
    result_image.SetOrigin(origin)
    itk.imwrite(result_image, output_path)



# 3. 数据预处理：调整分辨率、剪裁、标准化等
def preprocess_image(image_path, output_path, intensity_range=None):
    image = sitk.ReadImage(image_path)

    # 进行必要的预处理操作，如调整分辨率、剪裁等

    # 剪裁MR图像的强度值
    if intensity_range:
        clipped_image = sitk.Clamp(image, intensity_range[0], intensity_range[1])
    else:
        clipped_image = image

    # Z分数标准化
    normalized_image = sitk.Normalize(clipped_image)

    # 保存预处理后的图像
    sitk.WriteImage(normalized_image, output_path)


root_dir = "/home/chen/Documents/MR-CT_Synthesis_and_Segmentation/data/HaN-Seg/set_1"
for subdir in sorted(os.listdir(root_dir)):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        # 分别存储MR、CT和OAR图像的路径
        mr, ct, oars = None, None, []
        for filename in sorted(os.listdir(subdir_path)):
            filepath = os.path.join(subdir_path, filename)
            if 'MR_T1.NRRD' in filename.upper():
                mr = filepath
                mr_file = filename
            elif '_BED.NRRD' in filename.upper():
                ct = filepath
                ct_file = filename
            elif 'OAR' in filename.upper():
                oars.append(filepath)
        # 1. T2加权MR体积N4校正
        # if mr.endswith('MR_T1.nrrd'):
        #     print(mr)
        #     # 将原始文件名分为名字和扩展名
        #     base_name, ext = os.path.splitext(mr_file)
        #     # 在文件名后添加'remove_bed'标识
        #     n4_mr_savename = f"{base_name}_n4{ext}"
        #     n4_mr_savepath = os.path.join(subdir_path, n4_mr_savename)
        #     perform_n4_correction(mr, n4_mr_savepath)
        # print(ct)
        # 2. CT体积通过Elastix注册到T2加权MR体积
        if ct.endswith('bed.nrrd'):
            # 将原始文件名分为名字和扩展名
            base_name, ext = os.path.splitext(ct_file)
            # 在文件名后添加'remove_bed'标识
            ct_savename = f"{base_name}_register{ext}"
            ct_savepath = os.path.join(subdir_path, ct_savename)
            perform_image_registration(mr, ct, ct_savepath)