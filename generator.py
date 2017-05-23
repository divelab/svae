import os
from scipy.misc import imsave
import time
class Generator(object):
    
    def generate_and_save_images(self, nums, directory):
        t1= time.clock()
        imgs = self.sess.run(self.sample_out)
        t2= time.clock()
        print("image generation time is ==== %f"%(t2-t1) )
        for k in range(imgs.shape[0]):
            imgs_folder = os.path.join(directory, 'imgs_cleleba')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)
        #    imgs[k]= (imgs[k] + 1)*127.5
            imsave(os.path.join(imgs_folder,'%d.png') % k,
                imgs[k].reshape(self.height,self.width, 3))
 