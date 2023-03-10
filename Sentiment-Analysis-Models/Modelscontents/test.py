import Models as m

model_name = 'RF'
accuracy,precision,recall,f1 = m.Models("./data.csv",model_name)

print('accuracy:',accuracy)
print('precision:',precision)
print('recall:',recall)
print('f1:',f1)