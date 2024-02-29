import numpy as np

def discriminant_fun(x, mean, cov, prior):
    exponent = -0.5*np.dot(np.dot((x-mean).T,np.linalg.inv(cov)),(x-mean))
    return exponent-0.5*np.log(np.linalg.det(cov))+np.log(prior)

def predictions_Bayes(X, meanA, meanB, covA, covB, PA, PB):
    predic = []
    for i in X:
        classA = discriminant_fun(i, meanA, covA, PA)
        classB = discriminant_fun(i, meanB, covB, PB)
        if classA > classB:
            predic.append(0)
        else:
            predic.append(1)
            
    return np.array(predic)
    
def check_answer(y, predic):
    true_positive = np.sum((predic==1)&(y==1))
    true_negative = np.sum((predic==0)&(y==0))
    false_positive = np.sum((predic==1)&(y==0))
    false_negative = np.sum((predic==0)&(y==1))
    return true_positive, true_negative, false_positive, false_negative

def priors(classA, classB):
    P_classA = len(classA)/(len(classA)+len(classB))
    P_classB = len(classB)/(len(classA)+len(classB))
    return P_classA, P_classB

if __name__ == '__main__' :
    #set seed
    np.random.seed(1)

    # 1 Feature Class A 
    meanA_0 = np.array([3, 0])
    covA_0 = np.array([[0.10, 0],[0, 0.75]])
    sampleA_0 = 250

    # 1 FeatureClass B
    meanB_0 = np.array([6, 0])
    covB_0 = np.array([[0.10, 0],[0, 0.75]])
    sampleB_0 = 250

    # 100 Feature Class A 
    meanA_100 = np.zeros(100)
    meanA_100[0] = 3
    covA_100 = np.eye(100)*0.75
    covA_100[0][0] = 0.10
    sampleA_100 = 250

    # 100 Feature Class B
    meanB_100 = np.zeros(100)
    meanB_100[0] = 6
    covB_100 = np.eye(100)*0.75
    covB_100[0][0] = 0.10
    sampleB_100 = 250

    #sample data 1 Featurn
    classA_1data = np.random.multivariate_normal(meanA_0, covA_0, sampleA_0)
    classB_1data = np.random.multivariate_normal(meanB_0, covB_0, sampleB_0)

    #sample data 100 Featurn
    classA_100data = np.random.multivariate_normal(meanA_100, covA_100, sampleA_100)
    classB_100data = np.random.multivariate_normal(meanB_100, covB_100, sampleB_100)

    # X1 test, y1 check 
    X1 = np.vstack((classA_1data, classB_1data))
    y1 = np.hstack((np.zeros(sampleA_0), np.ones(sampleB_0)))

    # X100 test, y100 check 
    X100 = np.vstack((classA_100data, classB_100data))
    y100 = np.hstack((np.zeros(sampleA_100), np.ones(sampleB_100)))

    # Priors class
    P_A1, P_B1 = priors(classA_1data, classB_1data)
    P_A100, P_B100 = priors(classA_100data, classB_100data)

    # mean data class
    mean_classA_1 = np.mean(classA_1data, axis= 0)
    mean_classB_1 = np.mean(classB_1data, axis= 0)
    mean_classA_100 = np.mean(classA_100data, axis= 0)
    mean_classB_100 = np.mean(classB_100data, axis= 0)

    # cov data class
    cov_classA_1 = np.cov(classA_1data.T)
    cov_classB_1 = np.cov(classB_1data.T)
    cov_classA_100 = np.cov(classA_100data.T)
    cov_classB_100 = np.cov(classB_100data.T)

    #prediction
    predic_1 = predictions_Bayes(X1, mean_classA_1, mean_classB_1, cov_classA_1, cov_classB_1, P_A1, P_B1)
    predic_100 = predictions_Bayes(X100, mean_classA_100, mean_classB_100, cov_classA_100, cov_classB_100, P_A100, P_B100)

    #check answer
    T_P1, T_N1, F_P1, F_N1 = check_answer(y1, predic_1)
    T_P100, T_N100, F_P100, F_N100 = check_answer(y100, predic_1)

    #print cheak error
    print('_'*30)
    print('Featurn 1')
    print(f'true_positive = {T_P1}')
    print(f'true_negative = {T_N1}')
    print(f'false_positive = {F_P1}')
    print(f'true_negative = {F_N1}')
    print('_'*30)
    print('Featurn 100')
    print(f'true_positive = {T_P100}')
    print(f'true_negative = {T_N100}')
    print(f'false_positive = {F_P100}')
    print(f'true_negative = {F_N100}')