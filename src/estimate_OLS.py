import numpy as N

def estimate_OLS(desmtx,data,demean=1,resid=0):
    if demean == 1:      # use if desmtx doesn't include a constant
        data=data-N.mean(data)
    glm=N.linalg.lstsq(desmtx,data)
    if resid==1:  # return residuals as well
        return glm[0],glm[1]
    else:
        return glm[0]
