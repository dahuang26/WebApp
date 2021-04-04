from flask import Flask, render_template, request
import numpy as np
#import pandas as pd
#from scipy import interpolate
from scipy import linalg
from scipy.io import loadmat
from math import pi
from keras.models import load_model
#import matplotlib.pyplot as plt
#from geneticalgorithm import geneticalgorithm as ga
#import pickle as pk




#GA section
import sys
import time
from func_timeout import func_timeout, FunctionTimedOut
#import matplotlib.pyplot as plt

###############################################################################
###############################################################################
###############################################################################

class geneticalgorithm():
    
    '''  Genetic Algorithm (Elitist version) for Python
    
    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.
    
    
    
    Implementation and output:
        
        methods:
                run(): implements the genetic algorithm
                
        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }
            
                report: a list including the record of the progress of the
                algorithm over iterations
    '''
    #############################################################
    def __init__(self, function, dimension, variable_type='bool', \
                 variable_boundaries=None,\
                 variable_type_mixed=None, \
                 function_timeout=10,\
                 algorithm_parameters={'max_num_iteration': None,\
                                       'population_size':100,\
                                       'mutation_probability':0.1,\
                                       'elit_ratio': 0.01,\
                                       'crossover_probability': 0.5,\
                                       'parents_portion': 0.3,\
                                       'crossover_type':'uniform',\
                                       'max_iteration_without_improv':None},\
                     convergence_curve=True,\
                         progress_bar=True):


        '''
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function. 
        (For maximization multiply function by a negative sign: the absolute 
        value of the output would be the actual objective function)
        
        @param dimension <integer> - the number of decision variables
        
        @param variable_type <string> - 'bool' if all variables are Boolean; 
        'int' if all variables are integer; and 'real' if all variables are
        real value or continuous (for mixed type see @param variable_type_mixed)
        
        @param variable_boundaries <numpy array/None> - Default None; leave it 
        None if variable_type is 'bool'; otherwise provide an array of tuples 
        of length two as boundaries for each variable; 
        the length of the array must be equal dimension. For example, 
        np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first 
        and upper boundary 200 for second variable where dimension is 2.
        
        @param variable_type_mixed <numpy array/None> - Default None; leave it 
        None if all variables have the same type; otherwise this can be used to
        specify the type of each variable separately. For example if the first 
        variable is integer but the second one is real the input is: 
        np.array(['int'],['real']). NOTE: it does not accept 'bool'. If variable
        type is Boolean use 'int' and provide a boundary as [0,1] 
        in variable_boundaries. Also if variable_type_mixed is applied, 
        variable_boundaries has to be defined.
        
        @param function_timeout <float> - if the given function does not provide 
        output before function_timeout (unit is seconds) the algorithm raise error.
        For example, when there is an infinite loop in the given function. 
        
        @param algorithm_parameters:
            @ max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int> 
            @ mutation_probability <float in [0,1]>
            @ elit_ration <float in [0,1]>
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ crossover_type <string> - Default is 'uniform'; 'one_point' or 
            'two_point' are other options
            @ max_iteration_without_improv <int> - maximum number of 
            successive iterations without improvement. If None it is ineffective
        
        @param convergence_curve <True/False> - Plot the convergence curve or not
        Default is True.
        @progress_bar <True/False> - Show progress bar or not. Default is True.
        
        for more details and examples of implementation please visit:
            https://github.com/rmsolgi/geneticalgorithm
  
        '''
        self.__name__=geneticalgorithm
        #############################################################
        # input function
        assert (callable(function)),"function must be callable"     
        
        self.f=function
        #############################################################
        #dimension
        
        self.dim=int(dimension)
        
        #############################################################
        # input variable type
        
        assert(variable_type=='bool' or variable_type=='int' or\
               variable_type=='real'), \
               "\n variable_type must be 'bool', 'int', or 'real'"
       #############################################################
        # input variables' type (MIXED)     

        if variable_type_mixed is None:
            
            if variable_type=='real': 
                self.var_type=np.array([['real']]*self.dim)
            else:
                self.var_type=np.array([['int']]*self.dim)            

 
        else:
            assert (type(variable_type_mixed).__module__=='numpy'),\
            "\n variable_type must be numpy array"  
            assert (len(variable_type_mixed) == self.dim), \
            "\n variable_type must have a length equal dimension."       

            for i in variable_type_mixed:
                assert (i=='real' or i=='int'),\
                "\n variable_type_mixed is either 'int' or 'real' "+\
                "ex:['int','real','real']"+\
                "\n for 'boolean' use 'int' and specify boundary as [0,1]"
                

            self.var_type=variable_type_mixed
        #############################################################
        # input variables' boundaries 

            
        if variable_type!='bool' or type(variable_type_mixed).__module__=='numpy':
                       
            assert (type(variable_boundaries).__module__=='numpy'),\
            "\n variable_boundaries must be numpy array"
        
            assert (len(variable_boundaries)==self.dim),\
            "\n variable_boundaries must have a length equal dimension"        
        
        
            for i in variable_boundaries:
                assert (len(i) == 2), \
                "\n boundary for each variable must be a tuple of length two." 
                assert(i[0]<=i[1]),\
                "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound=variable_boundaries
        else:
            self.var_bound=np.array([[0,1]]*self.dim)
 
        ############################################################# 
        #Timeout
        self.funtimeout=float(function_timeout)
        ############################################################# 
        #convergence_curve
        if convergence_curve==True:
            self.convergence_curve=True
        else:
            self.convergence_curve=False
        ############################################################# 
        #progress_bar
        if progress_bar==True:
            self.progress_bar=True
        else:
            self.progress_bar=False
        ############################################################# 
        ############################################################# 
        # input algorithm's parameters
        
        self.param=algorithm_parameters
        
        self.pop_s=int(self.param['population_size'])
        
        assert (self.param['parents_portion']<=1\
                and self.param['parents_portion']>=0),\
        "parents_portion must be in range [0,1]" 
        
        self.par_s=int(self.param['parents_portion']*self.pop_s)
        trl=self.pop_s-self.par_s
        if trl % 2 != 0:
            self.par_s+=1
               
        self.prob_mut=self.param['mutation_probability']
        
        assert (self.prob_mut<=1 and self.prob_mut>=0), \
        "mutation_probability must be in range [0,1]"
        
        
        self.prob_cross=self.param['crossover_probability']
        assert (self.prob_cross<=1 and self.prob_cross>=0), \
        "mutation_probability must be in range [0,1]"
        
        assert (self.param['elit_ratio']<=1 and self.param['elit_ratio']>=0),\
        "elit_ratio must be in range [0,1]"                
        
        trl=self.pop_s*self.param['elit_ratio']
        if trl<1 and self.param['elit_ratio']>0:
            self.num_elit=1
        else:
            self.num_elit=int(trl)
            
        assert(self.par_s>=self.num_elit), \
        "\n number of parents must be greater than number of elits"
        
        if self.param['max_num_iteration']==None:
            self.iterate=0
            for i in range (0,self.dim):
                if self.var_type[i]=='int':
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*self.dim*(100/self.pop_s)
                else:
                    self.iterate+=(self.var_bound[i][1]-self.var_bound[i][0])*50*(100/self.pop_s)
            self.iterate=int(self.iterate)
            if (self.iterate*self.pop_s)>10000000:
                self.iterate=10000000/self.pop_s
        else:
            self.iterate=int(self.param['max_num_iteration'])
        
        self.c_type=self.param['crossover_type']
        assert (self.c_type=='uniform' or self.c_type=='one_point' or\
                self.c_type=='two_point'),\
        "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string" 
        
        
        self.stop_mniwi=False
        if self.param['max_iteration_without_improv']==None:
            self.mniwi=self.iterate+1
        else: 
            self.mniwi=int(self.param['max_iteration_without_improv'])

        
        ############################################################# 
    def run(self):
        
        
        ############################################################# 
        # Initial Population
        
        self.integers=np.where(self.var_type=='int')
        self.reals=np.where(self.var_type=='real')
        
        
        
        pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
        solo=np.zeros(self.dim+1)
        var=np.zeros(self.dim)       
        
        for p in range(0,self.pop_s):
         
            for i in self.integers[0]:
                var[i]=np.random.randint(self.var_bound[i][0],\
                        self.var_bound[i][1]+1)  
                solo[i]=var[i].copy()
            for i in self.reals[0]:
                var[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])    
                solo[i]=var[i].copy()


            obj=self.sim(var)            
            solo[self.dim]=obj
            pop[p]=solo.copy()

        #############################################################

        #############################################################
        # Report
        self.report=[]
        self.test_obj=obj
        self.best_variable=var.copy()
        self.best_function=obj
        ##############################################################   
                        
        t=1
        counter=0
        while t<=self.iterate:
            
            if self.progress_bar==True:
                self.progress(t,self.iterate,status="GA is running...")
            #############################################################
            #Sort
            pop = pop[pop[:,self.dim].argsort()]

                
            
            if pop[0,self.dim]<self.best_function:
                counter=0
                self.best_function=pop[0,self.dim].copy()
                self.best_variable=pop[0,: self.dim].copy()
            else:
                counter+=1
            #############################################################
            # Report

            self.report.append(pop[0,self.dim])
    
            ##############################################################         
            # Normalizing objective function 
            
            normobj=np.zeros(self.pop_s)
            
            minobj=pop[0,self.dim]
            if minobj<0:
                normobj=pop[:,self.dim]+abs(minobj)
                
            else:
                normobj=pop[:,self.dim].copy()
    
            maxnorm=np.amax(normobj)
            normobj=maxnorm-normobj+1

            #############################################################        
            # Calculate probability
            
            sum_normobj=np.sum(normobj)
            prob=np.zeros(self.pop_s)
            prob=normobj/sum_normobj
            cumprob=np.cumsum(prob)
  
            #############################################################        
            # Select parents
            par=np.array([np.zeros(self.dim+1)]*self.par_s)
            
            for k in range(0,self.num_elit):
                par[k]=pop[k].copy()
            for k in range(self.num_elit,self.par_s):
                index=np.searchsorted(cumprob,np.random.random())
                par[k]=pop[index].copy()
                
            ef_par_list=np.array([False]*self.par_s)
            par_count=0
            while par_count==0:
                for k in range(0,self.par_s):
                    if np.random.random()<=self.prob_cross:
                        ef_par_list[k]=True
                        par_count+=1
                 
            ef_par=par[ef_par_list].copy()
    
            #############################################################  
            #New generation
            pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            
            for k in range(0,self.par_s):
                pop[k]=par[k].copy()
                
            for k in range(self.par_s, self.pop_s, 2):
                r1=np.random.randint(0,par_count)
                r2=np.random.randint(0,par_count)
                pvar1=ef_par[r1,: self.dim].copy()
                pvar2=ef_par[r2,: self.dim].copy()
                
                ch=self.cross(pvar1,pvar2,self.c_type)
                ch1=ch[0].copy()
                ch2=ch[1].copy()
                
                ch1=self.mut(ch1)
                ch2=self.mutmidle(ch2,pvar1,pvar2)               
                solo[: self.dim]=ch1.copy()                
                obj=self.sim(ch1)
                solo[self.dim]=obj
                pop[k]=solo.copy()                
                solo[: self.dim]=ch2.copy()                
                obj=self.sim(ch2)               
                solo[self.dim]=obj
                pop[k+1]=solo.copy()
        #############################################################       
            t+=1
            if counter > self.mniwi:
                pop = pop[pop[:,self.dim].argsort()]
                if pop[0,self.dim]>=self.best_function:
                    t=self.iterate
                    if self.progress_bar==True:
                        self.progress(t,self.iterate,status="GA is running...")
                    time.sleep(2)
                    t+=1
                    self.stop_mniwi=True
                
        #############################################################
        #Sort
        pop = pop[pop[:,self.dim].argsort()]
        
        if pop[0,self.dim]<self.best_function:
                
            self.best_function=pop[0,self.dim].copy()
            self.best_variable=pop[0,: self.dim].copy()
        #############################################################
        # Report

        self.report.append(pop[0,self.dim])
        
        
 
        
        self.output_dict={'variable': self.best_variable, 'function':\
                          self.best_function}
        if self.progress_bar==True:
            show=' '*100
            sys.stdout.write('\r%s' % (show))
        sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        sys.stdout.flush() 
        re=np.array(self.report)
        if self.convergence_curve==True:
            # plt.plot(re)
            # plt.xlabel('Iteration')
            # plt.ylabel('Objective function')
            # plt.title('Genetic Algorithm')
            # plt.show()
            None
        if self.stop_mniwi==True:
            sys.stdout.write('\nWarning: GA is terminated due to the'+\
                             ' maximum number of iterations without improvement was met!')
##############################################################################         
##############################################################################         
    def cross(self,x,y,c_type):
         
        ofs1=x.copy()
        ofs2=y.copy()
        

        if c_type=='one_point':
            ran=np.random.randint(0,self.dim)
            for i in range(0,ran):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
  
        if c_type=='two_point':
                
            ran1=np.random.randint(0,self.dim)
            ran2=np.random.randint(ran1,self.dim)
                
            for i in range(ran1,ran2):
                ofs1[i]=y[i].copy()
                ofs2[i]=x[i].copy()
            
        if c_type=='uniform':
                
            for i in range(0, self.dim):
                ran=np.random.random()
                if ran <0.5:
                    ofs1[i]=y[i].copy()
                    ofs2[i]=x[i].copy() 
                   
        return np.array([ofs1,ofs2])
###############################################################################  
    
    def mut(self,x):
        
        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                
                x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1) 
                    
        

        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   

               x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0])    
            
        return x
###############################################################################
    def mutmidle(self, x, p1, p2):
        for i in self.integers[0]:
            ran=np.random.random()
            if ran < self.prob_mut:
                if p1[i]<p2[i]:
                    x[i]=np.random.randint(p1[i],p2[i])
                elif p1[i]>p2[i]:
                    x[i]=np.random.randint(p2[i],p1[i])
                else:
                    x[i]=np.random.randint(self.var_bound[i][0],\
                 self.var_bound[i][1]+1)
                        
        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   
                if p1[i]<p2[i]:
                    x[i]=p1[i]+np.random.random()*(p2[i]-p1[i])  
                elif p1[i]>p2[i]:
                    x[i]=p2[i]+np.random.random()*(p1[i]-p2[i])
                else:
                    x[i]=self.var_bound[i][0]+np.random.random()*\
                (self.var_bound[i][1]-self.var_bound[i][0]) 
        return x
###############################################################################     
    def evaluate(self):
        return self.f(self.temp)
###############################################################################    
    def sim(self,X):
        self.temp=X.copy()
        obj=None
        try:
            obj=func_timeout(self.funtimeout,self.evaluate)
        except FunctionTimedOut:
            print("given function is not applicable")
        assert (obj!=None), "After "+str(self.funtimeout)+" seconds delay "+\
                "func_timeout: the given function does not provide any output"
        return obj

###############################################################################
    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()     
###############################################################################            
###############################################################################
















app = Flask(__name__)

GM=loadmat('static\GM.mat')
cr=GM['cr']
#GA design parameters
Design=np.array([4, 6.5, 0.3, 31.4, 235,0.7])

#get model
def getmodel(N_st=None, h1=None, h2=None, b=None, E_clt=None, A_pt=None, E_pt=None, alpha=None, dy_pt=None, d_pt=None,\
             fy_steel=None, k0_ufp=None, ky_ufp=None, dy_ufp=None, n_ufp=None, F_pt_ini=None, Wall_width=None,n_pt=None):
    
    panel = {'t':None,'b':None,'h':None,'E':None,'G':None,'I':None,'H':None}
    PT={'a':None,'F_ini':None,'E':None,'l':None,'k_e':None,'k_y':None,'d_y':None,'d':None,'fy_steel':None,'n_pt':None}
    UFP={'n':None,'k_ini':None,'k_y':None,'d_y1':None}
    model={"panel":panel,'PT':PT,'UFP':UFP}

    H = h1 + (N_st - 1) * h2#total building height
    #get model porperties
    model['panel']['t']=b #wall thickness
    model['panel']['h']=H   # wall height
    model['panel']['b']=Wall_width # wall width
    model['panel']['E']=E_clt # wall elastic modulus
    model['panel']['G']=500 #wall shear modulus
    model['panel']['I']=model['panel']['t']*model['panel']['b']**3/12 #moment of inertia I=b*L^3/12
    model['panel']['H']=H
    
    model['PT']['n']=n_pt #number of PTbar
    model['PT']['a']=A_pt #area of PT bar
    model['PT']['fy_steel']=fy_steel #PTnumber
    model['PT']['F_ini']=F_pt_ini #initial force of PT
    model['PT']['E']=E_pt  #E pt
    model['PT']['l']=H #PT length
    model['PT']['k_e']=model['PT']['E']*model['PT']['a']/model['PT']['l']
    model['PT']['k_y']=alpha*model['PT']['E']*model['PT']['a']/model['PT']['l']
    model['PT']['d_y']=dy_pt #yielded displacement
    model['PT']['d']=d_pt #displacement to the rotational pivot point

    
    model['UFP']['n']=n_ufp   #number of UFP
    model['UFP']['k_ini']=k0_ufp    #initial stiffness 
    model['UFP']['k_y']=ky_ufp   #yielded stiffness
    model['UFP']['d_y1']=dy_ufp/Wall_width   #flag shape hysteresis's first yielded displacement

    return model

def getMK_Nst(model=None, N_st=None,M=None,h1=None,h2=None):

    DOF =N_st + 1
    h=np.zeros(N_st)
    for i in range (0,N_st):
        h[i]=h1+h2*i

    #get global m(mass) and k(stiffness)
    m =np.zeros([DOF, DOF])
    k =np.zeros([DOF, DOF])
    for i in range (0, N_st):
        m[i,i]=M
        m[i,N_st]=M*h[i]
        m[N_st,i]=M*h[i]
    
        if i == 0 and N_st != 1:
            k[i,i]=3*model['panel']['E']*model['panel']['I']/h[i]**3+3*model['panel']['E']*model['panel']['I']/(h[i+1]-h[i])**3
            k[i,i+1]=-3*model['panel']['E']*model['panel']['I']/(h[i+1]-h[i])**3
            k[i+1,i]=k[i,i+1]
        if i==N_st-1 and N_st !=1:
            k[i,i]=3*model['panel']['E']*model['panel']['I']/(h[i]-h[i-1])**3
            k[i,i+1]=0
            k[i+1,i]=k[i,i+1]
        if i!=0 and i!=N_st-1:
            k[i,i]=3*model['panel']['E']*model['panel']['I']/(h[i+1]-h[i])**3+3*model['panel']['E']*model['panel']['I']/(h[i]-h[i-1])**3
            k[i,i+1]=-3*model['panel']['E']*model['panel']['I']/(h[i+1]-h[i])**3
            k[i+1,i]=k[i,i+1]
      
    
    m_r = np.zeros([DOF, 1])
    for ii in range (0, N_st):
        m_r[ii,0]=M*h[ii]**2
  
    m[DOF-1, DOF-1]= sum(m_r)
    #k(DOF,DOF)=k_ro;
    m[DOF-1,0:-1]=0
    m[0:-1,DOF-1]=0
    
    return m,k,h





@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/submit', methods=['POST'])
def submit():
    Sd1=float(request.form['Sd1'])
    Sds=float(request.form['Sds'])
    N_st = int(request.form['story number'])
    w1 = float(request.form['story weight'])
    h1 = float(request.form['story height'])
    E_clt=float(request.form['E_clt'])
    dratio=float(request.form['dratio'])
    
    SLE_limit=float(request.form['SLE limit'])
    SLE_prob=float(request.form['SLE prob'])
    
    DBE_limit=float(request.form['DBE limit'])
    DBE_prob=float(request.form['DBE prob'])
    
    MCE_limit=float(request.form['MCE limit'])
    MCE_prob=float(request.form['MCE prob'])
    
    #m1=w1/386
    DOF=N_st+1

 
    g=386
    m1=w1/g
    T0=0.2*(Sd1/Sds)
    Ts=Sd1/Sds
    
    Target_SLE=np.array([SLE_limit,SLE_prob])
    Target_DBE=np.array([DBE_limit,DBE_prob])
    Target_MCE=np.array([MCE_limit,MCE_prob])
    
    #building Height [in]
    H=h1*N_st
    #aspect ratio of rocking wall
    AR=Design[0]
    #thickness of rocking wall  [in]
    t=Design[1]
    #F_pt_ini/F_pt_yield
    r=Design[2]
    #single Area of PT [in2]
    A_pt=Design[3]/8
    #single initial stiffness of UFP kips/in
    k0_ufp=Design[4]
    #yielding deformation of ufp [in]
    dy_ufp=Design[5]
    #damping ratio of building 
    dratio=0.05
    #Sa@0.1s [g]
    Sa_1=np.zeros(10)
    for i in range(1,11):
        Sa_1[i-1]=cr['Sa'][0,i-1][0,int(0.1/0.004)]
    #Sa@0.5s [g]
    Sa_2=np.zeros(10)
    for i in range(1,11):
        Sa_2[i-1]=cr['Sa'][0,i-1][0,int(0.5/0.004)]
    #Sa@1s [g]
    Sa_3=np.zeros(10)
    for i in range(1,11):
        Sa_3[i-1]=cr['Sa'][0,i-1][0,int(1/0.004)]
    #Sa@2s [g]
    Sa_4=np.zeros(10)
    for i in range(1,11):
        Sa_4[i-1]=cr['Sa'][0,i-1][0,int(2/0.004)]
    #Sa@3s [g]
    Sa_5=np.zeros(10)
    for i in range(1,11):
        Sa_5[i-1]=cr['Sa'][0,i-1][0,int(3/0.004)]
    #Sa@Tn [g]
    Sa_T=np.zeros(10)
    
    #additional parameters
    #Elastic modulus of steel [ksi]
    E_steel=29000
    #steel hardening parameters
    alpha=0.000904
    #yield stress of steel ksi
    fy_steel=105
    #rocking wall length  [in]
    b=h1*N_st/AR
    #yield deformation of PT
    dy_pt=0.1
    #number of PT
    n_pt=8
    #number of UFP
    n_ufp=1
    #yield force of PT
    Fy_pt=fy_steel*A_pt/n_pt
    #initial force of PT
    F_pt_ini=r*Fy_pt
    
    #assemble ANN input 
    range_inputs=np.array([[1,18],[0.05,2],[96,300],[1000,3600],[1,10],[3,25]\
                           ,[0.05,0.9],[5,50],[6.9,2188],[0.0006,700],[0.1,4],[0.01,0.2]])
    
    y_pred=np.zeros(10).T
    y_pred=np.reshape(y_pred,(10,1))
    
    
    p_SLE=np.zeros(10).T
    p_SLE=np.reshape(p_SLE,(10,1))
    p_DBE=np.zeros(10).T
    p_DBE=np.reshape(p_DBE,(10,1))
    p_MCE=np.zeros(10).T
    p_MCE=np.reshape(p_MCE,(10,1))
    
    price=np.array([0.0111,0.11])
    
     
    PGA_DBE=np.zeros(10)
    PGA_SLE=np.zeros(10)
    PGA_MCE=np.zeros(10)
    
    def f(Design=None):
        #aspect ratio of rocking wall
        AR=Design[0]
        #thickness of rocking wall  [in]
        t=Design[1]
        #F_pt_ini/F_pt_yield
        r=Design[2]
        #single Area of PT [in2]
        A_pt=Design[3]/8
        #single initial stiffness of UFP kips/in
        k0_ufp=Design[4]
        #yielding deformation of ufp [in]
        dy_ufp=Design[5]
        #rocking wall length  [in]
        b=h1*N_st/AR
        #yield deformation of PT
        dy_pt=0.1
        #number of PT
        n_pt=8
        #number of UFP
        n_ufp=1
        #yield force of PT
        Fy_pt=fy_steel*A_pt/n_pt
        #initial force of PT
        F_pt_ini=r*Fy_pt
        
    
        model=getmodel(N_st, h1, h1, t, E_clt, A_pt, E_steel, alpha, dy_pt, b/2,\
                  fy_steel, k0_ufp, k0_ufp*0.01, dy_ufp, n_ufp,F_pt_ini, b,n_pt)
        m,k,h=getMK_Nst(model, N_st,m1,h1,h1)
        k_ro=4*model['panel']['E']*model['panel']['I']/h1 
        k0=k.copy()
        k0[DOF-1,DOF-1]=k_ro
        # caluclate Tn
        Omega2=linalg.eigvals(k0, m)
        Tn=2*pi/(Omega2**(1/2))
        Tn=np.amax(Tn.real)
        
        #calculate Sa values
        for i in range(1,11):
            Sa_T[i-1]=cr['Sa'][0,i-1][0,int(Tn/0.004)]
            
        if Tn<T0:
            Sa=Sds*(0.4+0.6*Tn/T0)
        if Tn>T0 and Tn<Ts:
            Sa=Sds
        if Tn>Ts:
            Sa=Sd1/Tn
        if Tn<2:
            model_ANN = load_model('static/Saved ANN for Tn 0-2s/model_ANN.h5')
        if Tn>2:
            model_ANN = load_model('static/Saved ANN for Tn 2-3s/model_ANN.h5')
         #calculate Scale factor for DBE level
        SF=np.zeros(10)
    
        for i in range(1,11):
            SF[i-1]=Sa/Sa_T[i-1]
            PGA_DBE[i-1]=SF[i-1]*cr['pga_x'][0,i-1]
        PGA_SLE=PGA_DBE*0.5
        PGA_MCE=PGA_DBE*1.5   
        
        x_SLE=np.zeros((10,18))
        x_DBE=np.zeros((10,18))
        x_MCE=np.zeros((10,18))
        for i in range(10):
            x_SLE[i]=np.array([N_st,m1,h1,E_clt,AR,t,r\
                        ,t,k0_ufp,dy_ufp,PGA_SLE[i],dratio,Sa_1[i],\
                            Sa_2[i],Sa_3[i],Sa_4[i],Sa_5[i],Sa_T[i]])
            x_DBE[i]=np.array([N_st,m1,h1,E_clt,AR,t,r\
                   ,t,k0_ufp,dy_ufp,PGA_DBE[i],dratio,Sa_1[i],\
                       Sa_2[i],Sa_3[i],Sa_4[i],Sa_5[i],Sa_T[i]])
            x_MCE[i]=np.array([N_st,m1,h1,E_clt,AR,t,r\
                   ,t,k0_ufp,dy_ufp,PGA_MCE[i],dratio,Sa_1[i],\
                       Sa_2[i],Sa_3[i],Sa_4[i],Sa_5[i],Sa_T[i]])
        
        x_SLE=np.reshape(x_SLE,(10,18))
        x_DBE=np.reshape(x_DBE,(10,18))
        x_MCE=np.reshape(x_MCE,(10,18))
        for j in range(12):
            x_SLE[:,j]=(x_SLE[:,j]-range_inputs[j,0])/(range_inputs[j,1]-range_inputs[j,0])
            x_DBE[:,j]=(x_DBE[:,j]-range_inputs[j,0])/(range_inputs[j,1]-range_inputs[j,0])
            x_MCE[:,j]=(x_MCE[:,j]-range_inputs[j,0])/(range_inputs[j,1]-range_inputs[j,0])
     
        y_pred_SLE = model_ANN.predict(x_SLE)
        y_pred_DBE = model_ANN.predict(x_DBE)
        y_pred_MCE = model_ANN.predict(x_MCE)
        
        for i in range(10):  
            if y_pred_SLE[i]< Target_SLE[0]:
                
                p_SLE[i]=1
            else:
                p_SLE[i]=0
            
        Prob_SLE=np.sum(p_SLE)/10
       
        if Prob_SLE < Target_SLE[1]:
            Score_SLE=np.zeros(1)
            Score_SLE=np.inf
        else:
            h=h1*N_st
            V_clt=h/Design[0]*Design[1]*h
            V_steel=h*Design[3]
            Score_SLE=price[0]*V_clt+price[1]*V_steel
        if Tn>3:
            Score_SLE=np.inf
        
        for i in range(10):  
            if y_pred_DBE[i]< Target_DBE[0]:
                
                p_DBE[i]=1
            else:
                p_DBE[i]=0
            
        Prob_DBE=np.sum(p_DBE)/10
       
        if Prob_DBE < Target_DBE[1]:
            Score_DBE=np.zeros(1)
            Score_DBE=np.inf
        else:
            h=h1*N_st
            V_clt=h/Design[0]*Design[1]*h
            V_steel=h*Design[3]
            Score_DBE=price[0]*V_clt+price[1]*V_steel
        if Tn>3:
            Score_DBE=np.inf
        
        for i in range(10):  
            if y_pred_MCE[i]< Target_MCE[0]:
                
                p_MCE[i]=1
            else:
                p_MCE[i]=0
            
        Prob_MCE=np.sum(p_MCE)/10
       
        if Prob_MCE < Target_MCE[1]:
            Score_MCE=np.zeros(1)
            Score_MCE=np.inf
        else:
            h=h1*N_st
            V_clt=h/Design[0]*Design[1]*h
            V_steel=h*Design[3]
            Score_MCE=price[0]*V_clt+price[1]*V_steel
        if Tn>3:
            Score_MCE=np.inf
    
        Score=(Score_SLE+Score_DBE+Score_MCE)
    
        return Score
    
    

    varbound=np.array([[1,10],[5,25],[0.05,0.9],[5,50],[7,2188],[0.0006,10]])
    algorithm_param = {'max_num_iteration': 15,\
                        'population_size':100,\
                        'mutation_probability':0.1,\
                        'elit_ratio': 0.01,\
                        'crossover_probability': 0.5,\
                        'parents_portion': 0.3,\
                        'crossover_type':'uniform',\
                        'max_iteration_without_improv':None}
    
    model=geneticalgorithm(function=f,dimension=6,variable_type='real',variable_boundaries=varbound,algorithm_parameters=algorithm_param,function_timeout=6)
    model.run()
    solution=np.round(model.best_variable,2)

    return render_template('success.html', value=solution)
    

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True, use_reloader=False)