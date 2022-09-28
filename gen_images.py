import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt

class SLM_Image:

    def __init__(self):
        self.img_width = 3840
        self.xx = np.arange(0, self.img_width)/self.img_width
        self.modulator = 2*np.pi
        self.params = []
        
    def gen_bezier_image(self, x):
        """
        Convention - x is a list of 5 values
        grating order = x[0]
        x_pts = x[1],x[2] - np.array with values between [0,1)
        y_ pts = x[3],x[4] - np.array with values between [0,1)
        points_0 = (0,0)
        point_1 = (x[1],x[3])
        point_2 = (x[2],x[4])
        point_3 = (1,1)
        """

        if int(x[0]) == 0:
            pass
        else:
            grat_order = int(x[0])
            percent_width = float(1/np.abs(grat_order))
            T = int(percent_width * self.img_width)
            Imax = 1
            xpoints = np.array([x[1],x[2]])
            Ipoints = np.array([x[3], x[4]])

            T = int(percent_width * self.img_width)
            nrepeat_x = int(self.img_width/T)

            x1 = 0; I1 = 0
            x2 = xpoints[0]; I2 = Ipoints[0]
            x3 = xpoints[1]; I3 = Ipoints[1]
            x4 = 1; I4 = 1

            P1 = np.array((x1, I1)); P2 = np.array((x2, I2))
            P3 = np.array((x3, I3)); P4 = np.array((x4, I4))

            t = np.linspace(0, 1, T, endpoint=True)
            I = (1 - t) ** 3 * I1 + 3 * (1 - t) ** 2 * t * I2 + 3 * (1 - t) * t ** 2 * I3 + t ** 3 * I4
            
            img = np.tile(I, nrepeat_x + 10)
            img = img[:self.img_width]
            
            if grat_order < 0:
                img = 1 - img 
            else:
                img = img 
            return img 

    def gen_polynomial_image(self, x):
        """
        Convention - x is a list of 4 variables
        a = coeff_0.5 = x[0]
        b = coeff_1 = x[1]
        c = coeff_2 = x[1]
        modulator = x[3]
        """
        a = x[0]
        b = x[1]
        c = x[2]
        modulator = x[3]
        z_pts = (a * self.xx ** 0.5 + b * self.xx + c * self.xx ** 2) % modulator / modulator
        img = z_pts
        return img

if __name__ == '__main__':
    img_obj = SLM_Image()
    x = [-80, 0.25, 0.75, 0.25, 0.75] 
    img = img_obj.gen_bezier_image(x)
    x = [-300, -0, -0, 2*np.pi] 
    img = img_obj.gen_polynomial_image(x)
    plt.plot(img)
    plt.show()
    plt.savefig("test.png")
