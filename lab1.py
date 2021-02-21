from random import *
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

def gen_num():
    return randint(0,20)

print("\nМОПЕ Лаб.1 Рожко Михайло ІО-92\n")

a0 = gen_num()
a1 = gen_num()
a2 = gen_num()
a3 = gen_num()

all_x = [] #[x1,x2,x3],
for i in range(1,9):
    all_x.append([gen_num(),gen_num(),gen_num()])

all_y = []
for i in range(8):
    y = a0+a1*all_x[i][0]+a2*all_x[i][1]+a3*all_x[i][2]
    all_y.append(y)

all_x1, all_x2, all_x3 = [], [], []
for i in range(len(all_x)):
    all_x1.append(all_x[i][0])
    all_x2.append(all_x[i][1])
    all_x3.append(all_x[i][2])

x0_1 = (max(all_x1)+min(all_x1))/2
x0_2 = (max(all_x2)+min(all_x2))/2
x0_3 = (max(all_x3)+min(all_x3))/2

dx_1 = x0_1-min(all_x1)
dx_2 = x0_2-min(all_x2)
dx_3 = x0_3-min(all_x3)

x_n1, x_n2, x_n3 = [], [], []

for i in range(len(all_x)):
    x_n1.append(round(((all_x1[i] - x0_1) / dx_1), 4))
    x_n2.append(round(((all_x2[i] - x0_2) / dx_2), 4))
    x_n3.append(round(((all_x3[i] - x0_3) / dx_3), 4))

min_y = min(all_y)

data_main = [['', 'X1', 'X2', 'X3', 'Y', '', 'Xн1', 'Xн2', 'Xн3', '', 'min(Y)'],
             [1]+[i for i in all_x[0]]+[all_y[0]]+['']+[x_n1[0], x_n2[0], x_n3[0]]+['', min_y],
             [2]+[i for i in all_x[1]]+[all_y[1]]+['']+[x_n1[1], x_n2[1], x_n3[1]]+['', ''],
            [3]+[i for i in all_x[2]]+[all_y[2]]+['']+[x_n1[2], x_n2[2], x_n3[2]]+['', ''],
            [4]+[i for i in all_x[3]]+[all_y[3]]+['']+[x_n1[3], x_n2[3], x_n3[3]]+['', ''],
            [5]+[i for i in all_x[4]]+[all_y[4]]+['']+[x_n1[4], x_n2[4], x_n3[4]]+['', ''],
            [6]+[i for i in all_x[5]]+[all_y[5]]+['']+[x_n1[5], x_n2[5], x_n3[5]]+['', ''],
            [7]+[i for i in all_x[6]]+[all_y[6]]+['']+[x_n1[6], x_n2[6], x_n3[6]]+['', ''],
            [8]+[i for i in all_x[7]]+[all_y[7]]+['']+[x_n1[7], x_n2[7], x_n3[7]]+['', ''],
             ['x0', x0_1, x0_2, x0_3],
             ['dx',dx_1, dx_2, dx_3],]
data_a = [['a0', 'a1', 'a2', 'a3'],
          [a0, a1, a2, a3]]

table1 = ff.create_table(data_main)
table2 = ff.create_table(data_a)

fig = make_subplots(rows=2,
                          cols=1,
                          print_grid=True,
                          vertical_spacing=0.085,
                          subplot_titles=('', ''))

fig.add_trace(table1.data[0], 1, 1)
fig.add_trace(table2.data[0], 2, 1)

fig.layout.xaxis.update(table1.layout.xaxis)
fig.layout.yaxis.update(table1.layout.yaxis)
fig.layout.xaxis2.update(table2.layout.xaxis)
fig.layout.yaxis2.update(table2.layout.yaxis)

for k in range(len(table2.layout.annotations)):
        table2.layout.annotations[k].update(xref='x2', yref='y2')
all_annots = fig.layout.annotations+table1.layout.annotations + table2.layout.annotations
fig.layout.annotations = all_annots

fig.layout.update(width=800, height=600, margin=dict(t=100, l=50, r=50, b=50));
fig.show()




