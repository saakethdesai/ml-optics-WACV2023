import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from gen_images import SLM_Image

img_obj = SLM_Image()

database = []
#1D opt database
x = [-80, 0.25, 0.75, 0.25, 0.75] 
#for T in np.linspace(-81, 81, 1):
for T in np.arange(-80, 81, 2):
    for i in np.linspace(0, 1, 5): 
        for j in np.linspace(0, 1, 5): 
            for k in np.linspace(0, 1, 5): 
                for l in np.linspace(0, 1, 5):
                    #print (i, j, k, l, T)
                    if (T == 0):
                        pass
                    else:
                        x[0] = T
                        x[1] = i
                        x[2] = j
                        x[3] = k
                        x[4] = l
                        img = img_obj.gen_bezier_image(x)
                        database.append(img)
'''
#2D opt database
x = [0, 0, 0, 2*np.pi] 
for a1 in np.linspace(-800, 800, 50):
    for a2 in np.linspace(-800, 800, 40):
        x[1] = a1
        x[2] = a2
        img = img_obj.gen_polynomial_image(x)
        database.append(img)

#bezier database
x = [-80, 0.25, 0.75, 0.25, 0.75] 
for a1 in np.linspace(0, 1, 10):
    for a2 in np.linspace(0, 1, 10):
        for T in np.linspace(-81, 81, 20):
            if (T == 0):
                pass
            else:
                x[0] = T
                img = img_obj.gen_bezier_image(x)
                database.append(img)

#polynomial database
x = [0, 0, 0, 2*np.pi] 
for a0 in np.linspace(-500, 500, 10):
    for a1 in np.linspace(-800, 800, 10):
        for a2 in np.linspace(-800, 800, 10):
            x[0] = a0
            x[1] = a1
            x[2] = a2
            img = img_obj.gen_polynomial_image(x)
            database.append(img)
'''

database = np.array(database)
print (database.shape)
np.savetxt("database.txt", database)

#x = [-300, -0, -0, 2*np.pi] 
#img = img_obj.gen_polynomial_image(x)
#plt.plot(img)
#plt.show()
#plt.savefig("test.png")
