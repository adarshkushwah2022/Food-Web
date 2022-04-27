import pandas as pd
from scipy import spatial
from itertools import islice

"""**Importing dataset**"""

df = pd.read_csv('E:/IR Project Source Codes/Datasets/indian_food_Final.csv')

df.head()

df.columns

"""**Pre-processing Data**"""

#Converting each dish ingrdients string to list form
ingreList = list()
ingre = df['ingredients']
for x in ingre:
  l = list(x.split(","))
  for x1 in range(len(l)):
    l[x1]=l[x1].lstrip()
    l[x1]=l[x1].rstrip()
    l[x1] = l[x1].lower()
    l[x1]=l[x1].replace(" ", "")
  ingreList.append(l)

df.drop("ingredients", axis=1, inplace=True)

df['ingredients'] = ingreList

df.head()

dishNames = df['name']
recipeList = df['Recipe']

recipeDict={}
for x in range(len(dishNames)):
  recipeDict[dishNames[x]]=recipeList[x]


'''
Defining driver function:
Steps:
1. Taking input ingredients
2. Create input ingredients list
3. Pre-process the input ingredients entered
4. Calculating similarity between input ingredients list and ingredients list of each dish
   by using count vectorization method and storing the similarity value in dictionary.
5. Sorting dictionary on the basis on similarity value.
6. Fetching the top 5 entries of the dictionary
'''

def findTop5DishWithRecipe(inputI, N):
  inputI = str(inputI)
  input2 = inputI.split(",")
  for x1 in range(len(input2)):
    input2[x1]=input2[x1].lstrip()
    input2[x1]=input2[x1].rstrip()
    input2[x1]=input2[x1].replace(" ", "")

  for i1 in range(len(input2)):
    input2[i1]=input2[i1].lower()
    
  d1=dict()
  for index, row in df.iterrows():
    counter=0
    targetIngre = row['ingredients']
    for i in input2:
      if i in targetIngre:
        counter+=1
      d1[row['name']]=counter

  d2 = dict(sorted(d1.items(), key=lambda item: item[1],reverse = True))
  n_items=[]
  count2 = 0
  for x,y in d2.items():
    if(count2 == N):
      break
    n_items.append((x,y))
    count2= count2+1
    
  return n_items,len(input2)


def workerTwo(ingrdients, N):
  finalDishes = {}
  N=int(N)
  topDishes,N = findTop5DishWithRecipe(ingrdients, N)
  isAnyDishFound=False
  isPrinted=False
  for dishes in topDishes:
    if(dishes[1]>(N/2.0)):
      isAnyDishFound=True
      if(isPrinted == False):
        isPrinted = True
      finalDishes[dishes[0]]=recipeDict[dishes[0]]
  if(isAnyDishFound == False):
    finalDishes["No suitable dish found for the group of ingredients entered."]="Please try with different group of ingredients"
  return finalDishes
