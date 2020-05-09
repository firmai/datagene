## Additional
#========================================
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import numpy as np

def pil_to_cv_4d(gramian_img_steps_feat_4c):
  gramian_img_cv_steps_feat_4c = cv2.cvtColor(np.array(gramian_img_steps_feat_4c)[:,:,:3], cv2.COLOR_RGB2RGBA)
  gramian_img_cv_steps_feat_4c[:, :, 3] = np.array(gramian_img_steps_feat_4c)[:,:,-1]
  return gramian_img_cv_steps_feat_4c


def array_2d_to_grey_image(arr_2d, dim = 120 ):
  img_2c = Image.fromarray(arr_2d,mode="P")
  img_2c = img_2c.resize((dim, dim), Image.ANTIALIAS)
  return img_2c

  
def multi_plot(images, titles, title):
  fig = plt.figure(figsize=(8, 8))
  grid = ImageGrid(fig, 111,
                  nrows_ncols=(int(len(images)/2), 2),
                  axes_pad=0.3,
                  share_all=True,
                  cbar_location="right",
                  cbar_mode="single",
                  cbar_size="7%",
                  cbar_pad=0.3,
                  )

  for image, title, ax in zip(images, titles, grid):
      im = ax.imshow(image, cmap='rainbow', origin='lower')
      ax.set_title(title, fontdict={'fontsize': 12})
  ax.cax.colorbar(im)
  ax.cax.toggle_label(True)
  plt.suptitle(title, y=0.98, fontsize=16)
  plt.show()


def array_3d_to_rgb_image(gramian_fss_3d,dim=120):
  gramian_img_fss_3c_t = Image.fromarray(gramian_fss_3d, mode="RGB")
  gramian_img_fss_3c_t = gramian_img_fss_3c_t.resize((dim, dim), Image.ANTIALIAS)
  return gramian_img_fss_3c_t

def gram_array_3d_to_rgb_image(gramian_fss_3d,gramian_sff_3d, dim=120):
  gramian_img_fss_3c_t = array_3d_to_rgb_image(gramian_fss_3d.T, dim=dim)
  gramian_img_fss_3c = array_3d_to_rgb_image(gramian_fss_3d, dim=dim)
  gramian_img_sff_3c_t = array_3d_to_rgb_image(gramian_sff_3d, dim=dim)
  gramian_3d_img = Image.fromarray(np.hstack((np.array(gramian_img_fss_3c_t),np.array(gramian_img_fss_3c), np.array(gramian_img_sff_3c_t))))

  return gramian_3d_img

def array_4d_to_rgba_image(mpr_isff_4d,swap=(0,0), transpose=False, dim=120):
  if transpose:
    mpr_isff_4d = mpr_isff_4d.T
  mpr_img_isff_4c = Image.fromarray(mpr_isff_4d.swapaxes(swap[0],swap[1]), mode="RGBA")
  mpr_img_isff_4c = mpr_img_isff_4c.resize((dim, dim), Image.ANTIALIAS)
  return mpr_img_isff_4c

def three_plots(img_list,list_names):
  f,ax = plt.subplots(1,3)
  for en, imga,title in zip(range(3), img_list,list_names ):
      ax[en].imshow(imga)
      ax[en].set_title(title, fontdict={'fontsize': 12})

