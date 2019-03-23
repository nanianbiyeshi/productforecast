
# coding: utf-8

# In[1]:


import pandas as p
from sklearn.externals.six import StringIO
from sklearn import tree
import pydot
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


#数据量太大所以读取部分数据
limit=100000000
train =p.read_csv("D:/train_ver2.csv",error_bad_lines=False,dtype={"sexo":str, "ind_nuevo":str, "ult_fec_cli_1t":str,"indext":str},nrows=limit)
uid   = p.Series(train["ncodpers"].unique())
limit_people = 35000
uid  = uid.sample(n=limit_people)
train = train[train.ncodpers.isin(uid)]
test=p.read_csv("D:/test_ver2.csv",error_bad_lines=False)


# In[3]:


train.head(5)


# In[4]:


test.head(5)


# In[5]:


train.describe()


# In[6]:


test.describe()


# In[7]:


#日期格式转换
train["fecha_dato"] = p.to_datetime(train["fecha_dato"],format="%Y-%m-%d")
train["fecha_alta"] = p.to_datetime(train["fecha_alta"],format="%Y-%m-%d")
train["fecha_dato"].unique()


# In[8]:


#将age转化为数字类型
train["age"]   = p.to_numeric(train["age"], errors="coerce")
#检查那些列存在缺失值
test.isnull().sum()


# In[9]:


train.isnull().sum()


# In[10]:


#数据清洁
with sns.plotting_context('notebook',font_scale=1.5):
    sns.set_style("whitegrid")
    sns.distplot(train["age"].dropna(),
                 bins=80,
                 kde=False,
                 color="green")
    plt.title("Age Distribution")
    plt.ylabel("Count")


# In[11]:


#处理age列的无效值 将远离正常数据范围的数据用距离最近的均值代替
train.loc[train.age < 19,"age"]  = train.loc[(train.age >= 18) & (train.age <= 30),"age"].mean(skipna=True)
train.loc[train.age > 100,"age"] = train.loc[(train.age >= 31) & (train.age <= 100),"age"].mean(skipna=True)
train["age"].fillna(train["age"].mean(),inplace=True)
train["age"]= train["age"].astype(int)


# In[12]:


#处理ind_nuevo的缺失值
mon = train.loc[train["ind_nuevo"].isnull(),:].groupby("ncodpers", sort=False).size()
mon.max()


# In[13]:


#因此根据说明将缺失的ind_nuevo替换为1
train.loc[train["ind_nuevo"].isnull(),"ind_nuevo"] = 1
train.loc[train["ind_nuevo"].isnull(),"ind_nuevo"] = 1
#处理antiguedad
train.antiguedad = p.to_numeric(train.antiguedad,errors="coerce")
train.loc[train.antiguedad.isnull(),"antiguedad"] = train.antiguedad.min()
train.loc[train.antiguedad <0, "antiguedad"]  = 0
#处理nomprov
train.loc[train.nomprov.isnull(),"nomprov"] = "UNKNOWN"
#处理ind_nomina_ult1和ind_nom_pens_ult1
train.loc[train.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0
train.loc[train.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0


# In[14]:


#处理renta
g = train.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()
nin  = p.merge(train,g,how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]
nin   = nin.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")
train.sort_values("nomprov",inplace=True)
train = train.reset_index()
nin = nin.reset_index()
train.loc[train.renta.isnull(),"renta"] = nin.loc[train.renta.isnull(),"renta"].reset_index()
train.loc[train.renta.isnull(),"renta"] = train.loc[train.renta.notnull(),"renta"].median()
#处理test de renta 
test.loc[test.renta=='         NA',"renta"]=np.nan
g = test.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()
nin  = p.merge(test,g,how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]
nin   = nin.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")
test.sort_values("nomprov",inplace=True)
test = test.reset_index()
nin = nin.reset_index()
test.loc[test.renta.isnull(),"renta"] = nin.loc[test.renta.isnull(),"renta"].reset_index()
test.loc[test.renta.isnull(),"renta"] = test.loc[test.renta.notnull(),"renta"].median()


# In[15]:


#fecha_alta
dat=train.loc[:,"fecha_alta"].sort_values().reset_index()
da = int(np.median(dat.index.values))
train.loc[train.fecha_alta.isnull(),"fecha_alta"] = dat.loc[da,"fecha_alta"]
#
train.loc[train.indrel.isnull(),"indrel"] = 1
#
train.loc[train.indfall.isnull(),"indfall"] = "N"
train.loc[train.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
test.loc[test.tiprel_1mes.isnull(),"tiprel_1mes"] = "A"
train.tiprel_1mes = train.tiprel_1mes.astype("category")
test.tiprel_1mes=test.tiprel_1mes.astype("category")
#
map_dict = { 1.0  : "1",
            "1.0" : "1",
            "1"   : "1",
            "3.0" : "3",
            "P"   : "5",
            3.0   : "3",
            2.0   : "2",
            "3"   : "3",
            "2.0" : "2",
            "4.0" : "4",
            "4"   : "4",
            "2"   : "2"}
train.indrel_1mes.fillna("P",inplace=True)
train.indrel_1mes = train.indrel_1mes.apply(lambda x: map_dict.get(x,x))
#test
test.indrel_1mes.fillna("P",inplace=True)
test.indrel_1mes = test.indrel_1mes.apply(lambda x: map_dict.get(x,x))

#处理"tipodom","cod_prov
train.drop(["tipodom","cod_prov"],axis=1,inplace=True)
test.drop(["tipodom","cod_prov"],axis=1,inplace=True)


# In[16]:


train.loc[train.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = train["ind_actividad_cliente"].median()
#处理其余缺失数据 train
string_d = train.select_dtypes(include=["object"])
missing_columns = [col for col in string_d if string_d[col].isnull().any()]
unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
for col in unknown_cols:
    train.loc[train[col].isnull(),col] = "UNKNOWN"
#test    
string_d = test.select_dtypes(include=["object"])
missing_columns = [col for col in string_d if string_d[col].isnull().any()]
unknown_cols = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]
for col in unknown_cols:
    test.loc[test[col].isnull(),col] = "UNKNOWN"


# In[17]:


train.isnull().sum()


# 特征工程

# In[25]:


#将列标检转化为int
feature_cols = train.iloc[:1,].filter(regex="ind_+.*ult.*").columns.values
for col in feature_cols:
    train[col] = train[col].astype(int)


# In[26]:


train.dtypes


# In[28]:


test.isnull().sum()


# In[37]:


#非数字特征编码 
cloums=['fecha_dato', 'ind_empleado', 'pais_residencia','sexo', 'fecha_alta','ult_fec_cli_1t','tiprel_1mes', 'indresi', 'indext',
       'conyuemp', 'canal_entrada', 'indfall', 'nomprov','segmento']

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

encoder_x=LabelEncoder()
#train
for c in cloums:
    print(c)
    train[c]=encoder_x.fit_transform(train[c])
#test
for c in cloums:
    test[c]=encoder_x.fit_transform(test[c])


# In[60]:


#转化indrel_1mes、ind_nuevo为数字类型
train['ind_nuevo']=train['ind_nuevo'].astype(int)
train['indrel_1mes']=train['indrel_1mes'].astype(int)
#test
test['ind_nuevo']=test['ind_nuevo'].astype(int)
test['indrel_1mes']=test['indrel_1mes'].astype(int)
test['renta']=test['renta'].astype(float)
#展示编码后的特征数据
train.iloc[:,1:23].head(5)


# In[97]:


#数据归一化
from sklearn.preprocessing import Normalizer
train_x=Normalizer().fit_transform(train.iloc[:,1:23])
test_x=Normalizer().fit_transform(test.iloc[:,1:])
#换分标签
train_y=np.array(train.iloc[:,23:])


# In[58]:


import xgboost as xgb


# In[91]:


#模型训练函数
def runXGB(train_X,train_y, test_X, index, seed_val):
 
    train_index= index
    X_train = train_X[train_index]
    y_train = train_y[train_index]
 
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgtest  = xgb.DMatrix(test_X)
 
    param = {
        'objective' : 'multi:softprob',
        'eval_metric' : "mlogloss",
        'num_class' : 24,
        'silent' : 1,
        'min_child_weight' : 2,
        'eta': 0.05,
        'max_depth': 6,
        'subsample' : 0.9,
        'colsample_bytree' : 0.8,
        'seed' : seed_val
    }
    num_rounds = 100
    model  = xgb.train(param, xgtrain, num_rounds)
    pred   = model.predict(xgtest)
    return pred


# In[112]:


nfolds=5
seed_val=123
#模型训练
kf = KFold(train_x.shape[0], n_folds = nfolds, shuffle = True, random_state = seed_val)
preds = [0] * 24
for i, index in enumerate(kf):
         print(index)
         preds += runXGB(train_x, train_y, test_x, index, seed_val)
         print( 'fold %d' % (i + 1))
preds = preds / nfolds


# In[116]:


pre=p.DataFrame(preds,columns=train.iloc[:,23:].columns.tolist())


# In[117]:


#展示预测结果数值代表购买该产品的可能性
pre

