import sys
import numpy as np
from scipy import stats

# read x file and format the gender field
def read_x_file(file):
    data = np.genfromtxt(file, delimiter=',', dtype='str')
    formated_data = []
    for i in range(len(data)):
        g = data[i][0]
        if g == 'M':
            gender = np.array([1.0, 0.0, 0.0])
        elif g == 'F':
            gender = np.array([0.0, 1.0, 0.0])
        elif g == 'I':
            gender = np.array([0.0, 0.0, 1.0])
        dataWithoutGenderAsLetter = np.delete(data[i], 0).astype(np.float)
        dataWithoutGenderAsLetter = np.concatenate((gender, dataWithoutGenderAsLetter))
        formated_data.append(list(dataWithoutGenderAsLetter))
    return formated_data

#calculate z score normalization
def z_score(data,std,mean):
    j_range = len(data[0])
    for i in range(len(data)):
        for j in range(j_range):
            data[i][j] = (data[i][j] - mean[j]) / std[j]
    return data        

#normalize data using 
def normalize(train,test):
    std = np.std(train, axis = 0)
    mean = np.mean(train, axis = 0) 
    
    train = z_score(train,std,mean)
    test = z_score(test,std,mean)
    #train2 = stats.zscore(train)

#basic classifier methods and hyper params
class classifier():
    def __init__(self, data,labels,epochs,lr,k):
        self.data = data
        self.data_size = len(data)
        self.labels = labels
        self.epochs = epochs
        self.lr = lr
        self.k = k
        self.features = len(data[0])
        self.W = np.zeros((self.k, self.features))
    
    def predict(self,x):
        return np.argmax(np.dot(self.W, x))

######################
#MultiClass PERCEPTRON
######################
class perceptron():
    def __init__(self,data,labels,epochs,lr,k):
        self.classifier = classifier(data,labels,epochs,lr,k)
    
    def train(self):
        print("####PERCEPTRON####")
        for epoch in range(self.classifier.epochs):
            bad = 0
            for i in range(self.classifier.data_size):
                x=np.array(self.classifier.data[i])
                y = int(self.classifier.labels[i])
                y_hat = int(self.classifier.predict(x))
                
                W = self.classifier.W
                
                if y != y_hat:
                    bad += 1
                    self.classifier.W[y] += self.classifier.lr *x
                    self.classifier.W[y_hat] -= self.classifier.lr*x
            print("Epoch:",str(epoch),"; Accuracy:",str((self.classifier.data_size - bad)/self.classifier.data_size))

######################
#MultiClass SVM
######################
class svm():
    def __init__(self,data,labels,epochs,lr,k,lamda):
        self.classifier = classifier(data,labels,epochs,lr,k)
        self.lamda = lamda
        
    def train(self):
        print("####SVM####")
        for epoch in range(self.classifier.epochs):
            bad = 0
            for i in range(self.classifier.data_size):
                x=np.array(self.classifier.data[i])
                y = int(self.classifier.labels[i])
                y_hat = int(self.classifier.predict(x))
                
                self.classifier.W *= 1- self.classifier.lr*self.lamda
                
                if y != y_hat:
                    bad += 1
                    self.classifier.W[y] += self.classifier.lr *x
                    self.classifier.W[y_hat] -= self.classifier.lr*x
            print("Epoch:",str(epoch),"; Accuracy:",str((self.classifier.data_size - bad)/self.classifier.data_size))


######################
#MultiClass PassiveAggressive
######################
class pa():
    def __init__(self,data,labels,epochs,lr,k):
        self.classifier = classifier(data,labels,epochs,lr,k)
        
    def train(self):
        print("####PA####")
        for epoch in range(self.classifier.epochs):
            bad = 0
            for i in range(self.classifier.data_size):
                x=np.array(self.classifier.data[i])
                y = int(self.classifier.labels[i])
                y_hat = int(self.classifier.predict(x))
                
                if y != y_hat:
                    bad += 1
                    loss = max(0,1 - np.dot(self.classifier.W[y],x) + np.dot(self.classifier.W[y_hat],x))
                    tau = loss/(2 * (np.linalg.norm(x, ord=2) ** 2))
                    self.classifier.W[y] += tau *x
                    self.classifier.W[y_hat] -= tau*x
            print("Epoch:",str(epoch),"; Accuracy:",str((self.classifier.data_size - bad)/self.classifier.data_size))
                    

def main():
    train_file = sys.argv[1]
    train_label_file = sys.argv[2]
    test_file = sys.argv[3] if len(sys.argv) > 3 else False
    
    #train_file = "train_x.txt"
    #train_label_file = "train_y.txt"
    #test_file= "test_x.txt"
    
    #read files
    train = read_x_file(train_file)
    labels = np.genfromtxt(train_label_file,delimiter=',', dtype='str')
    test = read_x_file(test_file)
    
    normalize(train,test)
    model_perc = perceptron(train,labels,10,0.1,3)
    model_perc.train()
    
    model_svm = svm(train,labels,10,0.01,3,0.00001)
    model_svm.train()
    
    model_pa = pa(train,labels,20,0.1,3)
    model_pa.train()
    
    
    for x in test:
        print(f"perceptron: {model_perc.classifier.predict(x)}, svm: {model_svm.classifier.predict(x)}, pa: {model_pa.classifier.predict(x)}")
    
    
if __name__ == "__main__":
    main()