import pandas as pd
import numpy as np
#Neste caso, utilizarei o modelo VECM para as projeções
from statsmodels.tsa.vector_ar.vecm import VECM

class VECM_Model:
    def __init__(self, df, target, index, diff, coint, deterministic):
        self.df = df
        self.target = target
        self.index = index
        self.diff = diff
        self.coint = coint
        self.trend = deterministic
        self.model = None
        
    #Método que instancia o modelo de projeção
    def fit_model(self):
        
        #Instanciando modelo
        model = VECM(self.df, k_ar_diff=self.diff, coint_rank=self.coint, deterministic=self.trend)
        self.model = model.fit()

    def predict_model(self, pfrente):
        #Efetua a projeção
        predict = self.model.predict(steps=pfrente)
        
        #Pegando valores gerados pelo modelo
        array_predict = np.empty(pfrente, dtype = float)
        array_datas = np.empty(pfrente, dtype = 'datetime64[D]')

        #Iterando para prencher o DataFrame
        for i in range(pfrente):
        
            array_predict[i] = predict[i,self.index].round(2)

        for i in range(pfrente):
            self.df["Ano-Mês"] = self.df.index
            self.df["Ano-Mês"] = pd.to_datetime(self.df["Ano-Mês"])
            prox_data = self.df["Ano-Mês"].iloc[-1] + pd.DateOffset(months=i+1)
            array_datas[i] = str(prox_data)

        #Concatenando os DFs
        proj = pd.DataFrame({"Ano-Mês": array_datas,
                            self.target : array_predict})
        
        data = pd.concat([self.df, proj], ignore_index = False)
        data.set_index('Ano-Mês', inplace=True)

        return data