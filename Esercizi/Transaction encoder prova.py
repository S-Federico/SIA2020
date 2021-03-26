dataset=[['Milk','Onion','Nutmeg','Kidney Beans','Eggs','Yogurt'],
         ['Dill','Onion','Nutmeg','Kidney Beans','Eggs','Yogurt'],
         ['Milk','Apple','Kidney Beans','Eggs'],
         ['Milk','Unicorn','Corn','Kidney Beans','Yogurt'],
         ['Corn','Onion','Onion','Kidney Beans','Ice Cream','Eggs']]

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary= te.fit(dataset).transform(dataset)
df=pd.DataFrame(te_ary,columns=te.columns_)
print(df)

from mlxtend.frequent_patterns import apriori

apriori(df,min_support=0.6)

apriori(df,min_support=0.6,use_colnames=True)

frequent_itemsets=apriori(df,min_support=0.6,use_colnames=True)
frequent_itemsets['lenght']=frequent_itemsets['itemsets'].apply(lambda x:len(x))
print(frequent_itemsets)

frequent_itemsets[(frequent_itemsets['lenght']==2)&(frequent_itemsets['support']<=0.8)]

frequent_itemsets[frequent_itemsets['itemsets']=={'Onion','Eggs'}]

print(frequent_itemsets)