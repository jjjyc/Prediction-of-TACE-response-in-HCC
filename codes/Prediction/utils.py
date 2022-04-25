import shutil
import os

def check_create_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def spaser_pots(x, pots, x_min, x_max):
    leng = len(pots)
#    xd = np.array([0 for x in range(leng+2)])
    xd = [0 for x in range(leng+2)]
    if x >= pots[-1]:
        mid = (pots[-1] + x_max)/2.0
        dura = x_max - pots[-1]
        xd[-1] = (x - mid)/dura
        xd[-2] = 1
    else:
        for i in range(leng):
            if x < pots[i]:
                xd[i] = 1
                if i == 0:
                    mid = (pots[i] + x_min)/2.0
                    dura =  pots[i] - x_min
                else:
                    mid = (pots[i] + pots[i-1])/2.0
                    dura = pots[i] - pots[i-1]
                xd[-1] = (x - mid)/dura
                break
            
    return xd

    
def Discretization(x, feature_name, x_min, x_max):
    continuous = ["drug2", "drug-new","serum", "AST", "ALT", "CPR", "CP", "age", "ALB", "PT", "post_AST", "post_ALT", "post_CPR", "post_AFP", "ALT_change", "AST_change", "crp_change", "AFP"]
    if feature_name in continuous:
        if feature_name in ["drug2", "drug-new"]:
            pots = [5, 10, 15, 20, 30]
            
        if feature_name in ["serum"]:
            pots = [17, 34.2]
            
        if feature_name in ["AST", "post_AST"]:
            pots = [40, 61, 75]
            
        if feature_name in ["ALT", "post_ALT"]:
            pots = [40, 55, 73]
            
        if feature_name in ["CP", "post_CPR"]:
            pots = [5, 6, 7]
            
        if feature_name in ["CPR"]:
            pots = [1, 8]
            
        if feature_name in ["age"]:
            pots = [45, 55, 65]

        if feature_name in ["ALB"]:
            pots = [30, 35]

        if feature_name in ["PT"]:
            pots = [13, 16]

        if feature_name in ["post_AFP", "AFP"]:
            pots = [20, 200, 400]

        if feature_name in ["ALT_change"]:
            pots = [-0.06, 0.592, 6.92]

        if feature_name in ["AST_change"]:
            pots = [-0.065, 0.495, 6.057]

        if feature_name in ["crp_change"]:
            pots = [-0.043, 6.232]
            
#        input_channel = len(pots)+2
        xd = spaser_pots(x, pots, x_min, x_max)
    elif feature_name in ["HB","gender"]:
            xd = [x]
    return xd
    
    
    
