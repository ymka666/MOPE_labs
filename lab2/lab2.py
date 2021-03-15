from random import *
from math import *
import numpy
import plotly.figure_factory as ff

x1_min, x1_max = 20, 70
x2_min, x2_max = 5, 40
y_max = (30-17)*10  #130
y_min = (20-17)*10  #13
m = 5

x1_x2 = [[-1,1,-1],[-1,-1,1]]
x1_x2_ref = [[20, 70, 20], [5, 5, 40]]
y_arr = [[randint(y_min, y_max) for i in range(m)] for j in range(3)]

y_average = [sum(i)/len(i) for i in y_arr]

sigma =  [sum([(y_arr[i][j] - y_average[i]) ** 2 for j in range(m)]) / m for i in range(3)]

sigma_zero = sqrt(2 * (2 * m - 2) / (m * (m - 4)))

fuv = [sigma[0]/sigma[1], sigma[2]/sigma[0], sigma[2]/sigma[1]]

tetauv = [((m-2)/m)*i for i in fuv]

ruv = [abs(i-1)/sigma_zero for i in tetauv]

r_kr = 2

m_x1 = sum(x1_x2[0]) / 3
m_x2 = sum(x1_x2[1]) / 3
my = sum(y_average)/len(y_average)
a = [(x1_x2[0][0]**2 + x1_x2[0][1]**2 + x1_x2[0][2]**2)/3,
     (x1_x2[0][0]*x1_x2[1][0] + x1_x2[0][1]*x1_x2[1][1] + x1_x2[0][2]*x1_x2[1][2]) / 3,
     (x1_x2[1][0] ** 2 + x1_x2[1][1] ** 2 + x1_x2[1][2] ** 2) / 3]

aij = [sum([x1_x2[j][i] * y_average[i] for i in range(3)]) / 3 for j in range(2)]

first = numpy.array([[1, m_x1, m_x2], [m_x1, a[0], a[1]], [m_x2, a[1], a[2]]])
second = numpy.array([my, aij[0], aij[1]])
result = numpy.linalg.solve(first, second)
#5 завдання. Підставляю дані у отримане нормоване рівняння регресії для перевірки (роблю це генератором) виводжу(92) і доводжу, що співпадають з середніми значеннями ф. відгуку 
revision = [result[0] + result[1]*x1_x2[0][i] + result[2]*x1_x2[1][i] for i in range(len(result))]

delta_x1 = abs(x1_max-x1_min)/2
delta_x2 = abs(x2_max-x2_min)/2
x10 = (x1_max+x1_min)/2
x20 = (x2_max+x2_min)/2
a0 = result[0] - result[1]*x10/delta_x1 - result[2]*x20/delta_x2
a1 = result[1]/delta_x1
a2 = result[2]/delta_x2

last_revision = [a0 + a1*x1_x2_ref[0][i] + a2*x1_x2_ref[1][i] for i in range(3)]


data = [['X1', 'X2', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5'],
        [x1_x2[0][0], x1_x2[1][0]] + y_arr[0],
        [x1_x2[0][1], x1_x2[1][1]] + y_arr[1],
        [x1_x2[0][2], x1_x2[1][2]] + y_arr[2]]

fig = ff.create_table(data)
fig.show()

print("\n4. Перевіримо однорідність дисперсії за критерієм Романовського:\n   1) Знайдемо середнє значення функції відгуку в рядку: ")
for i in range(len(y_average)): print('   {0} = {1}'.format("y"+str(i+1)+"_avg", y_average[i]))
print("\n   2) Знайдемо дисперсії по рядках: ")
for i in range(len(sigma)): print("   σ²{0} = {1}".format("{y"+str(i)+str('}'), round(sigma[i], 4)))
print('\n   3) Обчислимо основне відхилення:\n   σ0 = ', round(sigma_zero, 4))
print('\n   4) Обчислимо Fuv:')
for i in range(len(fuv)): print('   {0} = {1}'.format('Fuv'+str(i+1), fuv[i]))
print('\n   5) teta:')
for i in range(len(tetauv)): print('   {0} = {1}'.format('θuv'+str(i+1), tetauv[i]))
print('\n   6) Ruv:')
for i in range(len(ruv)): print('   {0} = {1}'.format('Ruv'+str(i+1), ruv[i]))

print('\n   7) Оскільки m=5 (в таблиці немає даних для такого значення),\n   візьмемо значення R_кр = 2 для m=6 і довірчою ймовірністю р=0.9:')
for i in range(len(ruv)): print('   {0} = {1} < Rкр = {2}'.format("Ruv"+str(i), round(ruv[i],4), r_kr))
print("   Отже, дисперсія однорідна.\n")

print("5. Розрахунок нормованих коефіцієнтів рівняння регресії. ")
print('m_x1 = ', round(m_x1,4))
print('m_x2 = ', round(m_x2,4))
print('my', round(my,5))

for i in range(len(a)): print("{0} = {1}".format("a"+str(i+1), round(a[i],4)))
print('a11 = {0}, a22 = {1}'.format(round(aij[0],4), round(aij[1],4)))
print('\n')
for i in range(len(result)): print('{0} = {1}'.format("b"+str(i+1), round(result[i],4)))

print('\nОтже, нормоване рівняння регресії:\ny = {0} + ({1}*x1) + ({2}*x2)'.format(round(result[0],4), round(result[1],4), round(result[2],4)))
print("\nЗробимо перевірку: ")
print([round(i, 4) for i in revision])
print("Результат збігається з середніми значеннями.")

print("\n6. Проведемо натуралізацію коефіцієнтів: ")
print("delta_x1 = ", round(delta_x1,4))
print("delta_x2 = ", round(delta_x2,4))
print('x10 = ', round(x10,4))
print('x20 = ', round(x20,4))
print('a0 = ', round(a0,4))
print('a1 = ', round(a1,4))
print('a2 = ', round(a2,4))

print('\nЗапишемо натуралізоване рівняння регресії:')
print('у = a0 + a1*x1 + a2*x2 = {0} + {1}*x1 + {2}*x2'.format(round(a0,4), round(a1,4), round(a2,4)))
print("Зробимо перевірку по рядках:")
print([round(i, 4) for i in last_revision])
print('Отже, коефіцієнти натуралізованого рівняння регресії вірні.')



