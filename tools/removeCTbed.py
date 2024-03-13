import pydicom
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import ndimage
import re
import glob
from PIL import Image
from skimage.morphology import label
from collections import OrderedDict
import skimage
from tqdm import tqdm, trange
import argparse
import nrrd
import multiprocessing as mp
import traceback


# 加载dicom
def load_scan(path, type):
    # print(path)
    if type == 'CT':
        slices = glob.glob(path + '/CT*.dcm')
    else:
        slices = glob.glob(path + '/*.dcm')
    dicoms = []
    for i in slices:
        s = pydicom.dcmread(i)
        dicoms.append(s)
    dicoms.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return dicoms


# 重采样+灰度转HU
def dicoms2npy_rs(dicoms):
    image = np.stack([s.pixel_array for s in dicoms])
    # 灰度 -> HU
    if dicoms[0].Modality == 'CT':
        for slice_number in range(len(dicoms)):
            intercept = dicoms[slice_number].RescaleIntercept  # 截距
            slope = dicoms[slice_number].RescaleSlope  # 斜率
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int32)
            image = image.astype(np.int16)
            image[slice_number] += np.int16(intercept)
    return image


def npy2dicoms(npy_image, dir):
    path1 = f'./dicom/{dir}/CT'  # 包含dcm文件的文件夹的路径
    if not os.path.exists(path1):
        print(f"can't find {path1} patient CT!")
        return

    path2 = f'./result/{dir}'
    if not os.path.exists(path2):
        os.makedirs(path2)

    for path_, _, file_ in os.walk(path1):
        L = len(file_)

        if L > 0:
            for f in sorted(file_, reverse=True):
                if "CT" not in f:
                    continue
                file1 = os.path.abspath(os.path.join(path_, f))
                image = pydicom.dcmread(file1)
                # data_img = Image.fromarray(image.pixel_array)
                sliceID = image.InstanceNumber - 1

                intercept = image.RescaleIntercept  # 截距
                slope = image.RescaleSlope  # 斜率
                if slope != 1:
                    npy_image[sliceID] = npy_image[sliceID].astype(np.float64)
                    npy_image[sliceID] = npy_image[sliceID] / slope
                    npy_image[sliceID] = npy_image[sliceID].astype(np.int16)
                npy_image[sliceID] -= np.int16(intercept)

                image.PixelData = npy_image[sliceID].tobytes()
                image.Rows, image.Columns = npy_image[sliceID].shape
                image.save_as(os.path.join(path2, f))
    return


# # 画ct分布
# def ct_distr(ctnpy):
#     mask = ctnpy <= -1000
#     ctnpy_mask = np.ma.array(ctnpy, mask=mask)
#     plt.hist(ctnpy_mask.flatten(), bins=80, color='c')
#     plt.xlabel("Hounsfield Units (HU)")
#     plt.ylabel("Frequency")
#     plt.savefig(f'./visual/distr/CT/{i}.jpg')
#     plt.close()
#
#
# # 画mr分布
# def mr_distr(mrnpy):
#     mask = mrnpy <= 0
#     mrnpy_mask = np.ma.array(mrnpy, mask=mask)
#     plt.hist(mrnpy_mask.flatten(), bins=80, color='c')
#     plt.xlabel("MR")
#     plt.ylabel("Frequency")
#     plt.savefig(f'./visual/distr/MR/{i}.jpg')
#     plt.close()


# 求一张二值图片的最大连通分量
def mcc(mask01):
    # 定义连通分量有序字典
    region_volume = OrderedDict()
    # 获取各连通分量map和个数
    mask_map, numregions = label(mask01 == 1, return_num=True)
    # 连通分量个数
    region_volume['num_region'] = numregions
    # 总点个数，容量
    total_volume = 0
    # 最大的区域容量
    max_region = 0
    # 最大区域的flag
    max_region_flag = 0
    # 枚举每个连通分量
    for l in range(1, numregions + 1):
        # 第l个连通分量的容量
        region_volume[l] = np.sum(mask_map == l)  # * volume_per_volume
        # 如果大于最大容量，则该连通分量为最大连通分量
        if region_volume[l] > max_region:
            max_region = region_volume[l]
            max_region_flag = l
        # 计算总容量
        total_volume += region_volume[l]
        # 最初的mask初始化
    maskmcc = mask01.copy()
    # 保留最大连通分量的点
    maskmcc[mask_map != max_region_flag] = 0
    maskmcc[mask_map == max_region_flag] = 1
    return maskmcc


# 去除主干之外的杂讯
def get_mask(a, patientid, j):
    mpl.use('TkAgg')  # !IMPORTANT 更改在这里！！！！！！！！！
    ct_np = a
    # 1二值化，生成mask
    mask = np.zeros_like(ct_np).astype(np.uint8)
    mask[ct_np > (-300 + 1000)] = 1
    if mask.max() == 0:
        return mask

    plt.subplot(221)
    plt.imshow(mask * 255, cmap='gray')

    # 2开运算 使狭窄的连接断开和消除细毛刺 消亮点亮条
    kernel = skimage.morphology.disk(1)
    mask = skimage.morphology.opening(mask, kernel)

    plt.subplot(222)
    plt.imshow(mask * 255, cmap='gray')

    # 3最大连通分量
    mask = mcc(mask)
    plt.subplot(223)
    plt.imshow(mask * 255, cmap='gray')

    # 4闭运算 弥合狭窄的间断， 填充小的孔洞 消暗点暗条
    kernel = skimage.morphology.disk(15)  # 圆核
    mask = skimage.morphology.closing(mask, kernel)

    contours = skimage.measure.find_contours(mask, 0.8)

    for n, contour in enumerate(contours):
        contour_coords = contour.round().astype(int)
        rr, cc = skimage.draw.polygon(contour_coords[:, 0], contour_coords[:, 1])
        mask[rr, cc] = 1

    plt.subplot(224)
    plt.imshow(mask * 255, cmap='gray')

    if not os.path.exists(f'./visual/mask/{patientid}'):
        os.makedirs(f'./visual/mask/{patientid}')
    if j % 10 == 0:
        plt.savefig(f'./visual/mask/{patientid}/{patientid}_{j}.jpg')
    plt.close()
    return mask


# 删除空白slice
def remove_empty_slices_and_bed(rsct, patientid):
    l = rsct.shape[0]
    ctnpy = []

    for j in range(l):
        if rsct[j].max() != 0:
            mask = get_mask(rsct[j], patientid, j)
            rsct[j] = mask * rsct[j]
            ctnpy.append(rsct[j])
        else:
            ctnpy.append(rsct[j])

    ctnpy = np.array(ctnpy).astype(np.int16)
    return ctnpy


def remove_background(rsct, rsmr):
    l = rsct.shape[0]
    a = []
    b = []
    for j in range(0, l):
        if rsmr[j].max() != 0 and rsct[j].max() != 0:
            a.append(rsct[j])
            b.append(rsmr[j])
    ctnpy = np.array(a).astype(np.int16)
    mrnpy = np.array(b).astype(np.int16)
    l = ctnpy.shape[1]
    a = []
    b = []
    for j in range(0, l):
        if mrnpy[:, j, :].max() != 0 and ctnpy[:, j, :].max() != 0:
            a.append(mrnpy[:, j, :])
            b.append(ctnpy[:, j, :])
    ctnpy = np.array(a).astype(np.int16)
    mrnpy = np.array(b).astype(np.int16)
    l = ctnpy.shape[2]
    a = []
    b = []
    for j in range(0, l):
        if mrnpy[:, :, j].max() != 0 and ctnpy[:, :, j].max() != 0:
            a.append(mrnpy[:, :, j])
            b.append(ctnpy[:, :, j])
    ctnpy = np.array(a).astype(np.int16).swapaxes(0, 2)
    mrnpy = np.array(b).astype(np.int16).swapaxes(0, 2)

    return ctnpy, mrnpy


# 画slices图
def plot_slices(ctnpy, patient_id, type):
    if not os.path.exists(f'./visual/plot/{type}/{patient_id}/'):
        os.makedirs(f'./visual/plot/{type}/{patient_id}/')
    for j, k in enumerate(ctnpy):
        if j % 10 == 0:
            plt.imshow(k, cmap=plt.cm.bone)
            plt.savefig(f'./visual/plot/{type}/{patient_id}/{patient_id}_{j}.jpg')
            plt.close()


def process_subdir(arg):
    import os
    import nrrd
    plt.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    subdir_path, counter = arg
    try:
        for filename in sorted(os.listdir(subdir_path)):
            filepath = os.path.join(subdir_path, filename)
            if 'CT.NRRD' in filename.upper():
                ct = filepath
                print(ct)
                if filename.endswith('.nrrd'):
                    # 将原始文件名分为名字和扩展名
                    base_name, ext = os.path.splitext(filename)
                    # 在文件名后添加'remove_bed'标识
                    new_filename = f"{base_name}_remove_bed{ext}"
                # 保存处理结果
                savePath = os.path.join(subdir_path, new_filename)
                if os.path.exists(savePath):
                    break;
                nrrd_data, nrrd_options = nrrd.read(ct)
                print(nrrd_data.shape)
                nrrd_array = np.asarray(nrrd_data)
                nrrd_array = np.swapaxes(nrrd_array, 0, 2)
                nrrd_array += 1000
                # 删除背板
                ctnpy = remove_empty_slices_and_bed(nrrd_array, counter)
                ctnpy = ctnpy - 1000
                # if args.draw:
                #     plot_slices(ctnpy, counter, 'CT')
                ctnpy = np.swapaxes(ctnpy, 0, 2)
                counter += 1
                new_filename = "remove_bed_" + filename
                # 检查是否有'.nrrd'扩展名，并在其前加上'remove_bed'

                print(savePath)
                nrrd.write(savePath, ctnpy, nrrd_options)
    except Exception as e:
        # 捕获所有异常，并将异常信息写入到文件中
        with open("error_log.txt", "a") as file:
            file.write(f"Error in process {mp.current_process().name}: {e}\n")
            file.write(traceback.format_exc())  # 这将写入完整的堆栈跟踪信息


def main(dataroot):
    subdir_paths = [(os.path.join(dataroot, subdir), count) for count, subdir in enumerate(sorted(os.listdir(dataroot)))
                    if os.path.isdir(os.path.join(dataroot, subdir))]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(process_subdir, subdir_paths)


if __name__ == "__main__":
    dataroot = '/home/chen/Documents/MR-CT_Synthesis_and_Segmentation/data/HaN-Seg/set_1/'
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--draw", type=bool, default=False)
    parser.add_argument("-p", "--position", type=int)
    args = parser.parse_args()
    main(dataroot)

# if __name__ == '__main__':
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     mpl.rcParams['axes.unicode_minus'] = False
#     dataroot = '/home/chen/Documents/MR-CT_Synthesis_and_Segmentation/data/HaN-Seg/set_1/'
#     patients = os.listdir(dataroot)
#     count = 0
#
#     for subdir in sorted(os.listdir(dataroot)):
#         subdir_path = os.path.join(dataroot, subdir)
#         if os.path.isdir(subdir_path):
#             # 分别存储MR、CT和OAR图像的路径
#             ct = None
#             counter = 0
#             for filename in sorted(os.listdir(subdir_path)):
#                 filepath = os.path.join(subdir_path, filename)
#                 if 'CT' in filename.upper():
#                     ct = filepath
#                 print(ct)
#                 nrrd_data, nrrd_options = nrrd.read(ct)
#                 print(nrrd_data.shape)
#                 nrrd_array = np.asarray(nrrd_data)
#                 nrrd_array = np.swapaxes(nrrd_array, 0, 2)
#                 nrrd_array += 1000
#                 # 删除背板
#                 ctnpy = remove_empty_slices_and_bed(nrrd_array, counter)
#                 ctnpy = ctnpy - 1000
#                 if args.draw:
#                     plot_slices(ctnpy, counter, 'CT')
#                 nrrd_array = np.swapaxes(nrrd_array, 0, 2)
#                 counter += 1
#                 # 保存处理结果
#                 if not os.path.exists("./nrrd/CT/"):
#                     os.makedirs("./nrrd/CT/")
#                 savePath = os.path.join("./nrrd/CT/", filename)
#                 nr.write(savePath, ctnpy, nrrd_options)
