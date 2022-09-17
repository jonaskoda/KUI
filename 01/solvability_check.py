from re import X
import npuzzle

def pos_of_x(matrix):
   for i, value in enumerate(matrix):
      if value == None:
         if ((i+1) // 4) == 0 or ((i+1) // 4) == 3:
            x = 1 #even
            return x    
         else:
            x = 0 #odd
            return x
 
def is_solvable(env):
   matrix = []
   #converting numbers into a matrix
   for i in range(4):
      for j in range(4):
         if env._NPuzzle__is_inside(i, j):
            x = env.read_tile(i, j)
            matrix.append(x)
   
   length = len(matrix)
   inversions = 0
   
   #calculating inversions
   for k in range(len(matrix)):
      for index, value in enumerate(matrix[k:]):
         if (value == None) or (matrix[k] == None):
            continue
         elif matrix[k] > value: 
            inversions += 1  
   #print(inversions)   
   #3x3
   if len(matrix) == 9:
      if inversions % 2 == 1:
         return False
      elif inversions %2 == 0:
         return True
   #4x4
   if len(matrix) == 16:
      if inversions % 2 == 1: #odd
         if pos_of_x(matrix) == 1: #even
            return True
         else:
            return False

      if inversions % 2 == 0: #even
         if pos_of_x(matrix) == 0: #odd
            return True
         else:
            return False


if __name__=="__main__":
   env = npuzzle.NPuzzle(4)
   env.reset()
   env.visualise()
   # just check
   print(is_solvable(env))
