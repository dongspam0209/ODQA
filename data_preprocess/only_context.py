import pandas as pd
data=pd.read_csv("/data/ephemeral/home/kdh/dataset/train.csv")
context_column=data['context']

print(context_column.head())

context_column.to_csv("/data/ephemeral/home/kdh/dataset/train_only_context.csv",encoding='utf-8-sig',index=False)