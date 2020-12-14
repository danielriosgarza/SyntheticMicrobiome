# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:41:26 2020

@author: u0139894
"""

import streamlit as stl

import numpy as np
import scipy.integrate as solver
import pandas as pd
import copy
import graphviz as graphviz



class Strain:
    '''
    Parameters of a single strain.
    
    name : strain id
    
    x_0 : initial abundance
    
    q_0 : initial lag phase parameter
    
    mu : max growth rate
    
    metab : ordered list of metab ids that are imported or secreted by the strain.
    
    metab_v : consumption/production of metabolites in the metab list.
             (positive - consumption; negative - production)
             
    metab_k : Monod constants for the metabolites in the metab list
    
    growth_model : composition of the growth rate term. Each tuple has a constant
                  and a list of metabolites. The constant and the monod equations for
                  each of the metabolites in a tuple are multipled. the final growth rate
                  consists of the sum of these products. An arbitrary number of terms can be 
                  used.
                  
    feeding : terms used in the importing/secreting of metabs. Binary indices that
             indicate which growth rate terms to use for each metabolite in the 
             metab list.
    
    monoculture: a strain object with the monoculture parameters (optional)
      
    '''
    def __init__(self, name, x_0, q_0, mu, metab, metab_v, metab_k, growth_model, feeding):
        
        self.name = name
        self.x_0 = x_0
        self.q_0 = q_0
        self.mu = mu
        
        self.metab = np.array(metab)
        self.metab_v = np.array(metab_v)
        self.metab_k = np.array(metab_k)
        self.growth_model = growth_model
        self.feeding = np.array([np.array(feeding[i])*self.metab_v[i] for i in range(len(self.metab))]) 
        
        self.monoculture=None
        
        
    
    
    
    
class Community:
    
    '''
    strains : list with defined 'strain' obj
    
    metabolome : list with the names of external metabolites
    
    metabolome_c : list with the concentration of external metabolites
    
    
    
    '''
    def __init__(self, strains, metabolome, metabolome_c):
        
        
        self.strain_obj = self.__take_strains(strains)
        self.nstrains = len(self.strain_obj)
        self.strains = np.array([i.name for i in strains])
        self.metabolome = np.array(metabolome)
        self.metabolome_c = np.array(metabolome_c)
        self.nmets = len(self.metabolome_c)
        self.ndims = self.nstrains + self.nmets
        self.metab_v, self.metab_k, self.mus, self.qs, self.x_0 = self.__parse_strains(self.strain_obj)
        self.feeding_m = self.__get_feeding_matrices(self.strain_obj)
        self.growth_rate_functions = [self.__make_growth_function(i) for i in self.strain_obj]
        self.community_dyn = None
        self.environment_dyn = None
        self.lag_dyn = None
        self.system_time = None
        self.feeding_process = {i:np.zeros(self.nmets) for i in self.strains}
    
    def __take_strains(self, strains):
        if len(strains)==1:
            if strains[0].monoculture is not None:
                return [strains[0].monoculture]
        return strains
    
    def __normalize_feeding_process(self, feeding_process):
        
        mult={i:{'pos':0, 'neg':0} for i in self.metabolome}
        for i in self.feeding_process:
            for z,v in enumerate(self.metabolome):
                if self.feeding_process[i][z]<0:
                    mult[v]['neg'] += np.round(np.abs(self.feeding_process[i][z]),4)
                elif self.feeding_process[i][z]>0:
                    mult[v]['pos'] += np.round(np.abs(self.feeding_process[i][z]),4)
        norm_f1 = copy.deepcopy(self.feeding_process)
        for i in self.feeding_process:
            for z,v in enumerate(self.metabolome):
                if self.feeding_process[i][z]<0:
                    norm_f1[i][z] = np.round(norm_f1[i][z],4)/mult[v]['neg']
                elif self.feeding_process[i][z]>0:
                    norm_f1[i][z] = np.round(norm_f1[i][z],4)/mult[v]['pos']
        return norm_f1
        
    
    def __parse_strains(self, strains_obj):
        
        
        
        metab_v= {i: np.zeros(self.nstrains) for i in self.metabolome}
        metab_k= {i: np.zeros(self.nstrains) for i in self.metabolome}
        
        mus = np.zeros(self.nstrains)
        qs = np.zeros(self.nstrains)
        x_0 = np.zeros(self.nstrains)
        
        
        for i,v in enumerate(strains_obj):
            
            mus[i] = v.mu
            qs[i] = v.q_0
            x_0[i] = v.x_0
            
            for z,v2 in enumerate(v.metab):
                if v2 in self.metabolome:
                    metab_v[v2][i],metab_k[v2][i]  = v.metab_v[z],v.metab_k[z]
        
        metab_v = np.array([metab_v[i] for i in self.metabolome])
        metab_k = np.array([metab_k[i] for i in self.metabolome])
        
        return metab_v, metab_k, mus, qs,x_0
    
    
    def __get_feeding_matrices (self, strains_obj):
        
        matrices=[]
        for i,v in enumerate(strains_obj):
            fm = np.zeros((self.nmets, len(v.feeding[0])))
            
            for z,m in enumerate(self.metabolome):
                if m in v.metab:
                    fm[z] = v.feeding[list(v.metab).index(m)]
            matrices.append(fm)
        
        return matrices
    
    def __make_growth_function(self, strain_obj):
        
        multipliers = []
        
        for i in strain_obj.growth_model:
            d=[i[0]]
            for z in i[1::]:
                if z in self.metabolome:
                    d.append((strain_obj.metab_k[list(strain_obj.metab).index(z)], list(self.metabolome).index(z)))
            multipliers.append(d)
    
        
        
        def growth_function(s):
            
            l=[]
            
            for i in multipliers:
                a = i[0]
                
                for z in i[1::]:
                    a*= s[z[1]]/(z[0] + s[z[1]])
                l +=[a]
            
            return np.array(l)
        return growth_function
    
            
                
        
        
        
        
    def lagPhase(self, q):
        """ 
        Lag phases for each species
        
        """
        return q/(1+q)
    
    def growthRates(self, s,q):
        """ 
        
        """
        lag = self.lagPhase(q)
        gr = []
        for i,v in enumerate(self.growth_rate_functions):
            gr.append(self.mus[i]*lag[i]*v(s))
            
            
        return gr
    
    def batch_dynamics(self, t,y):
        """ 
        Definitions of the ODE's 
        
        """
        
        
        x=y[0:self.nstrains]
        s=y[self.nstrains:self.ndims]
        q=y[self.ndims:]
        
        x=x*(x>0)
        s=s*(s>0)
        growth = self.growthRates(s,q)
        #print(growth)
        
        dyn_x = np.array([sum(i) for i in growth])*x   
        dyn_s = np.zeros(self.nmets)
        
        for i in range(self.nstrains):
            feedk=self.feeding_m[i].dot(growth[i]*x[i])
            dyn_s-=feedk
            self.feeding_process[self.strains[i]]-=feedk
        
        dyn_q = self.mus*q*(q<10.**5)    # Lag phase: ignore this once q > 10.**5
        
        dyn = np.append(dyn_x,np.append(dyn_s,dyn_q))
        
        return dyn
    
    def chemostat_dynamics(self, t,y, dilution=.1, feed=None):
        """ 
        Definitions of the ODE's 
        
        """
        
        
        if feed is None:
            feed=self.metabolome_c
        
        
        x=y[0:self.nstrains]
        
        s=y[self.nstrains:self.ndims]
        
        
        
        
        #s+=feed*dilution
        
        
        x=x*(x>0)
        s=s*(s>0)
        #print(s)
        q=y[self.ndims:]
        
        growth = self.growthRates(s,q)
        #print(growth)
        #s+=s*dilution
        dyn_x = (np.array([sum(i) for i in growth])*x)-dilution*x
        dyn_s = np.zeros(self.nmets)
        
        for i in range(self.nstrains):
            feedk=self.feeding_m[i].dot(growth[i]*x[i])
            
            self.feeding_process[self.strains[i]]-=feedk
            
            dyn_s-=feedk
            

        dyn_s = dyn_s - dilution*s + dilution*feed
        
        
        
        
        
        dyn_q = self.mus*q*(q<10.**5)    # Lag phase: ignore this once q > 10.**5
        
        dyn = np.append(dyn_x,np.append(dyn_s,dyn_q))
        
        return dyn
    
   

    def simulate(self, dynamics, t_start=0, t_end = 50, nsteps = 1000, method = 'bdf', dilution=None):
        """ 
        Simulate the ODE's 
        
        """
        # Solve ODE
        
        ode = solver.ode(dynamics)
        
        # BDF method suited to stiff systems of ODEs
        
        ode.set_integrator('vode',nsteps=nsteps,method= method)
        
        y_init = np.append(self.x_0, np.append(self.metabolome_c, self.qs))
        
        # Time
       
        t_step = (t_end - t_start)/nsteps
        
        if dilution is not None:
            ode.set_f_params(dilution)
        ode.set_initial_value(y_init,t_start)
        
        ts = []
        ys = []
        
        while ode.successful() and ode.t < t_end:
            ode.integrate(ode.t + t_step)
            ts.append(ode.t)
            ys.append(ode.y)
        
        time = np.array(ts)
        self.system_time=time
        y_total = np.vstack(ys).T    #y = (x,s,q)
        
        community_dyn = {}
        environment_dyn = {}
        lag_dyn={}
        
        for i,v in enumerate(self.strains):
            community_dyn[v] = y_total[i]
        
        for i in range(self.nmets):
            environment_dyn[self.metabolome[i]]=y_total[i+self.nstrains]
        for i,v in enumerate(self.strains):
            lag_dyn[v] = y_total[i+self.nstrains + self.nmets]
        
        self.community_dyn = pd.DataFrame.from_dict(community_dyn)
        self.community_dyn.index=time
        self.environment_dyn = pd.DataFrame.from_dict(environment_dyn)
        self.environment_dyn.index=time
        self.lag_dyn = pd.DataFrame.from_dict(lag_dyn)
        self.lag_dyn.index = time
        #update sharing of resources
        for i in self.feeding_process:
            self.feeding_process[i] = np.round(self.feeding_process[i]/self.system_time[-1],4)
    
    
    def PeriodicFeed_simulate(self, dynamics, t_start=0, t_end = 100, nsteps = 1000, method = 'bdf', feed_interval=1, dilution=.01):
        """ 
        Simulate the ODE's 
        
        """
        # Solve ODE
        ode = solver.ode(dynamics)
        
        # BDF method suited to stiff systems of ODEs
        
        ode.set_integrator('vode',nsteps=nsteps,method= method)
        
        y_init = np.append(self.x_0, np.append(self.metabolome_c, self.qs))
        
        # Time
       
        t_step = (t_end - t_start)/nsteps
        
        ode.set_initial_value(y_init,t_start)
        
        ts = [0]
        ys = [y_init]
        
        time = t_start
        btime=[]
        while ode.successful() and ode.t < t_end:
            if np.round(time)%feed_interval==0:
                if np.round(time) not in btime:
                    btime.append(np.round(time))
                    #print(time)
                    rout = np.append(np.ones(self.nstrains)*(1-dilution), np.append(np.ones(self.nmets)*(1-dilution), np.ones(self.nstrains)))
                    cy = (ys[-1]*rout) + np.append(np.zeros(self.nstrains), np.append(self.metabolome_c*dilution, np.zeros(self.nstrains)))
                    #print(btime)
                    ode.set_initial_value(cy, ode.t)
            time+=t_step
                
                
                
            ode.integrate(ode.t + t_step)
            ts.append(ode.t)
            ys.append(ode.y)
        
        time = np.array(ts)
        self.system_time=time
        y_total = np.vstack(ys).T    #y = (x,s,q)
        
        community_dyn = {}
        environment_dyn = {}
        lag_dyn={}
        
        for i,v in enumerate(self.strains):
            community_dyn[v] = y_total[i]
        
        for i in range(self.nmets):
            environment_dyn[self.metabolome[i]]=y_total[i+self.nstrains]
        for i,v in enumerate(self.strains):
            lag_dyn[v] = y_total[i+self.nstrains + self.nmets]
        
        self.community_dyn = pd.DataFrame.from_dict(community_dyn)
        self.community_dyn.index=time
        self.environment_dyn = pd.DataFrame.from_dict(environment_dyn)
        self.environment_dyn.index=time
        self.lag_dyn = pd.DataFrame.from_dict(lag_dyn)
        self.lag_dyn.index = time
        #update sharing of resources
        for i in self.feeding_process:
            self.feeding_process[i] = np.round(self.feeding_process[i]/self.system_time[-1],4)
    
    
    




###

stl.sidebar.write('### Initial Parameters')

####Create strain objects####

strains_in_community = []

# RI

if stl.sidebar.checkbox('R. intestinalis'):
    
    
    
    #initial density
    rid0 = stl.sidebar.slider('Initial density(RI)', 0.001, 100.0, 1.31, step=0.001, format='%.3f')
    x_0_RI = rid0
    
    
    #initial lagphase parameter
    rilp = stl.sidebar.slider('lagPhaseParm (RI)', 0.000, 1.000, 1.00, step=0.000001, format='%.6f')
    q_0_RI = rilp
    
    
    #max growth rate
    mu_RI =  2.125
    
    #weight for nutrient dependence
    w_RI = [1, 0.848]  
    
    #metabolites that are consumed or produced
    metabs_RI = ['Fructose', 'Formate', 'Acetate', 'Butyrate']
    
    #consumption (positive) or production(negative) rates for the above metabolites
    metabs_RI_v = [0.364, 0.012, 1.03, -0.528]
    
    #Monod cosntants for the cosumption of the above metabolites
    metabs_RI_k = [205., 0., 70.525, 0.]
    
    # terms for the growth rate consisting of additive terms 
    # with the general form: weight*[metabolite_1]*...*[metabolite_n]
    # the [metabolite] term is the Monod equation. Each tuple is 
    # an independent additive term. If known other terms can be added, 
    # as long the name, consuption/production constants, and Monod constants
    # are added in the above lists and later to the community object
    growth_RI = [(w_RI[0], 'Fructose'), (w_RI[1], 'Fructose', 'Acetate')]
    
    # which of the growth rate terms are used in the equations
    # for metabolite concentrations. For example, there are two 
    # growth terms (above) and four metabolites for the RI strain. 
    # Fructose, the first metabolite uses both growth terms (1,1) 
    # while Acetate only uses only the second term (0,1)
    feeding_RI = [(1,1), (1,1), (0,1), (1,1)]
    
    #A strain object is created
    RI = Strain('RI', x_0_RI, q_0_RI, mu_RI, metabs_RI, metabs_RI_v, metabs_RI_k, growth_RI, feeding_RI)
    #Because the monoculture parameters are different. Add a nested class with the monoculture values
    x_0_RI_mc = RI.x_0
    q_0_RI_mc = RI.q_0
    mu_RI_mc = 0.76
    w_RI_mc = [1, 1.081] 
    metabs_RI_mc = ['Fructose', 'Formate', 'Acetate', 'Butyrate']
    metabs_RI_v_mc = [0.576, 0.0, 1.093, -0.625]
    metabs_RI_k_mc = [80.065, 0., 104.625, 0.]
    growth_RI_mc = [(w_RI[0], 'Fructose'), (w_RI[1], 'Fructose', 'Acetate')]
    feeding_RI_mc = [(1,1), (1,1), (0,1), (1,1)]
    RI.monoculture = Strain('RI (mc)', x_0_RI_mc, q_0_RI_mc, mu_RI_mc, metabs_RI_mc, metabs_RI_v_mc, metabs_RI_k_mc, growth_RI_mc, feeding_RI_mc)

    
    
    strains_in_community.append(RI)
    
    stl.sidebar.write('*-----------------------------------------*\n\n\n')



# FP
if stl.sidebar.checkbox('F. prausnitzii'):

    
    
        
    fp = stl.sidebar.slider('Initial density (FP)', 0.001, 100.0, 0.11, step=0.001, format='%.3f')
    x_0_FP = fp
    
    rifp = stl.sidebar.slider('lagPhaseParm. (FP)', 0.000, 1.00, 10**(-5), step=0.000001, format='%.6f')
    q_0_FP = rifp
    
    mu_FP =  2.397
    w_FP = [1,0.237]  
    metabs_FP = ['Fructose', 'Formate', 'Acetate', 'Butyrate', 'unknown_compound']
    metabs_FP_v = [1.962, -1.684, 11.443, -2.263, 2.485]
    metabs_FP_k = [12., 0, 41., 0., 62.]
    growth_FP = [(w_FP[0], 'unknown_compound', 'Fructose'), (w_FP[1], 'unknown_compound', 'Fructose', 'Acetate')]
    feeding_FP = [(1,1), (1,1), (0,1), (1,1), (1,1)]
    
    FP = Strain('FP', x_0_FP, q_0_FP, mu_FP, metabs_FP, metabs_FP_v, metabs_FP_k, growth_FP, feeding_FP)
    x_0_FP_mc = FP.x_0
    q_0_FP_mc = FP.q_0
    mu_FP_mc =  2.4
    w_FP_mc = [1,0.237] 
    metabs_FP_mc = ['Fructose', 'Formate', 'Acetate', 'Butyrate', 'unknown_compound']
    metabs_FP_v_mc = [1.962, -1.684, 11.443, -2.263, 2.485]
    metabs_FP_k_mc = [12., 0, 41., 0., 62.]
    growth_FP_mc = [(w_FP[0], 'unknown_compound', 'Fructose'), (w_FP[1], 'unknown_compound', 'Fructose', 'Acetate')]
    feeding_FP_mc = [(1,1), (1,1), (0,1), (1,1), (1,1)]
    FP.monoculture = Strain('FP (mc)', x_0_FP_mc, q_0_FP_mc, mu_FP_mc, metabs_FP_mc, metabs_FP_v_mc, metabs_FP_k_mc, growth_FP_mc, feeding_FP_mc)


    strains_in_community.append(FP)
    stl.sidebar.write('*-----------------------------------------*\n\n\n')




# BH
if stl.sidebar.checkbox('B. hydrogenotrophica'):
    
    
    
    bh = stl.sidebar.slider('Initial density (BH)', 0.001, 100.0, 0.28, step=0.001, format='%.3f')
    bhlp = stl.sidebar.slider('lagPhaseParm.(BH)', 0.000, 1.000, 10**(-1), step=0.000001, format='%.6f')
    q_0_BH = bhlp
    x_0_BH = bh
    
    mu_BH =  1.823
    w_BH = [1.0, 1.0] 
    metabs_BH = ['Fructose', 'Formate', 'Acetate']
    metabs_BH_v = [0.389, 2.516, -0.617]
    metabs_BH_k = [93.661, 413., 0.]
    growth_BH = [(w_BH[0], 'Fructose'), (w_BH[1], 'Formate')]
    feeding_BH = [(1,0), (0,1), (1,1)]
    
    BH = Strain('BH', x_0_BH, q_0_BH, mu_BH, metabs_BH, metabs_BH_v, metabs_BH_k, growth_BH, feeding_BH)
    x_0_BH_mc = BH.x_0
    q_0_BH_mc = BH.q_0
    mu_BH_mc =  2.0
    w_BH_mc = [1.0, 1.0] 
    metabs_BH_mc = ['Fructose', 'Formate', 'Acetate']
    metabs_BH_v_mc = [3.0, 5.0, -0.6]
    metabs_BH_k_mc = [200, 200, 0.]
    growth_BH_mc = [(w_BH[0], 'Fructose'), (w_BH[1], 'Formate')]
    feeding_BH_mc = [(1,0), (0,1), (1,1)]

    BH.monoculture = Strain('BH (mc)', x_0_BH_mc, q_0_BH_mc, mu_BH_mc, metabs_BH_mc, metabs_BH_v_mc, metabs_BH_k_mc, growth_BH_mc, feeding_BH_mc)

    
    strains_in_community.append(BH)
    



#def build_graph(simulated_c_object):
    


# metabolites and concentration from the external environment
# Only the metabolites matching these will be modeled even if other metabolites are in
# the specific strains
metabolome = ['Fructose', 'Formate', 'Acetate', 'Butyrate', 'unknown_compound']

stl.sidebar.write('*-----------------------------------------*\n\n\n')
fructose = stl.sidebar.slider('Fructose', 0.0, 100.0, 47.4)
formate = stl.sidebar.slider('Formate', 0.0, 100.0, 1.13)
acetate = stl.sidebar.slider('Acetate', 0.0, 100.0, 4.19)
butyrate = stl.sidebar.slider('Butyrate', 0.0, 100.0, 1.78)
unkCompound = stl.sidebar.slider('unkCompound', 0.0, 100.0, 30.0)
metabolome_c = [fructose,formate,acetate,butyrate,unkCompound]
stl.sidebar.write('*-----------------------------------------*\n\n\n')
#create a community object
c = Community(strains_in_community, metabolome, metabolome_c)



#simualate

if stl.checkbox('PeriodicFeed'):
    try:
        interval = stl.slider('Feed interval', 1, 100, 12)
        dilution1 = stl.slider('Dilution', 0.0, 1.0, 0.25, step=0.0001, format='%.4f')
        simulTpf = stl.slider('SimulTime(p.f)', 0, 10000, 100)
        simul=c.PeriodicFeed_simulate(c.batch_dynamics, feed_interval=interval, dilution=dilution1, t_end = simulTpf)
        
        stl.write('### Community')
        
        stl.line_chart(c.community_dyn)
        stl.write('### Compounds')
        stl.line_chart(c.environment_dyn)
        
        #make crossfeeding graph
        stl.write('### Cross Feeding')
        graph = graphviz.Digraph()
        
        for name in c.strains:
            #graph.attr('node', shape='square')
            graph.node(name,label=name, **{'shape':'circle'})#width':str(i), 'height':str(i)}))
        
        for name in c.metabolome:
            graph.node(name, label=name, **{'shape':'rectangle', 'width':'2', 'height':'.5', 'labelfontsize':'1', 'fixedsize':'true'})
            
        norm_feed =  c._Community__normalize_feeding_process(c.feeding_process)
        print(norm_feed)
        
        for i, v in enumerate(c.strains):
            for i2,v2 in enumerate(c.metabolome):
                if norm_feed[v][i2]<0:
                    graph.edge(v2, v, **{'penwidth':str(2*abs(norm_feed[v][i2]))})
                elif norm_feed[v][i2]>0:
                    graph.edge(v, v2, **{'penwidth':str(2*abs(norm_feed[v][i2]))})
        
        stl.graphviz_chart(graph)
        
    except ValueError:
        stl.write('### Select a strain')
    
if stl.checkbox('Chemostat'):
    try:
        dilution2 = stl.slider('Dilution(Periodic)', 0.0, 1.0, 0.25, step=0.0001, format='%.4f')
        simulTC = stl.slider('SimulTime(chms)', 0, 10000, 100)
        simul=c.simulate(c.chemostat_dynamics, dilution=dilution2, t_end=simulTC)
    
        stl.write('### Community')
        stl.line_chart(c.community_dyn)
        stl.write('### Compounds')
        stl.line_chart(c.environment_dyn)
        
        #make crossfeeding graph
        stl.write('### Cross Feeding')
        graph = graphviz.Digraph()
        
        for name in c.strains:
            #graph.attr('node', shape='square')
            graph.node(name,label=name, **{'shape':'circle'})#width':str(i), 'height':str(i)}))
        
        for name in c.metabolome:
            graph.node(name, label=name, **{'shape':'rectangle', 'width':'2', 'height':'.5', 'labelfontsize':'1', 'fixedsize':'true'})
            
        norm_feed =  c._Community__normalize_feeding_process(c.feeding_process)
        print(norm_feed)
        
        for i, v in enumerate(c.strains):
            for i2,v2 in enumerate(c.metabolome):
                if norm_feed[v][i2]<0:
                    graph.edge(v2, v, **{'penwidth':str(2*abs(norm_feed[v][i2]))})
                elif norm_feed[v][i2]>0:
                    graph.edge(v, v2, **{'penwidth':str(2*abs(norm_feed[v][i2]))})
        
        stl.graphviz_chart(graph)
        
    except ValueError:
        stl.write('### Select a strain')
        
    
if stl.checkbox('Batch'):
    try:
        simulB = stl.slider('SimulTime(Batch)', 0, 10000, 100)
        simul=c.simulate(c.batch_dynamics, t_end=simulB)
    
        stl.write('### Community')
        stl.line_chart(c.community_dyn)
        stl.write('### Compounds')
        stl.line_chart(c.environment_dyn) 
        
        #make crossfeeding graph
        stl.write('### Cross Feeding')
        graph = graphviz.Digraph()
        
        for name in c.strains:
            #graph.attr('node', shape='square')
            graph.node(name,label=name, **{'shape':'circle'})#width':str(i), 'height':str(i)}))
        
        for name in c.metabolome:
            graph.node(name, label=name, **{'shape':'rectangle', 'width':'2', 'height':'.5', 'labelfontsize':'1', 'fixedsize':'true'})
            
        norm_feed =  c._Community__normalize_feeding_process(c.feeding_process)
        print(norm_feed)
        
        for i, v in enumerate(c.strains):
            for i2,v2 in enumerate(c.metabolome):
                if norm_feed[v][i2]<0:
                    graph.edge(v2, v, **{'penwidth':str(2*abs(norm_feed[v][i2]))})
                elif norm_feed[v][i2]>0:
                    graph.edge(v, v2, **{'penwidth':str(2*abs(norm_feed[v][i2]))})
        
        stl.graphviz_chart(graph)
        
        
        
    except ValueError:
        stl.write('### Select a strain')
        


